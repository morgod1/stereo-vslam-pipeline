"""Record a seeded, synchronous stereo sequence on a dedicated CARLA server."""

import argparse
import csv
import json
import platform
import queue
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import carla
from PIL import Image

from simulator.geometry import calibration, optical_pose, vehicle_pose


ROOT = Path(__file__).resolve().parents[1]


def save_rgb(image, path):
    # CARLA supplies BGRA bytes; discard alpha and explicitly preserve RGB order.
    rgb = Image.frombytes("RGB", (image.width, image.height), image.raw_data, "raw", "BGRX")
    rgb.save(path, compress_level=1)


def wait_for_frame(sensor_queue, frame, timeout=30.0):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for camera frame {frame}")
        try:
            image = sensor_queue.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError(f"Timed out waiting for camera frame {frame}") from error
        if image.frame > frame:
            raise RuntimeError(f"Missing camera frame {frame}; received {image.frame}")
        if image.frame == frame:
            return image


def make_route(world_map, location, config):
    waypoint = world_map.get_waypoint(location)
    route = []
    for _ in range(config["route_points"]):
        choices = waypoint.next(config["route_spacing_m"])
        if not choices:
            break
        waypoint = min(choices, key=lambda item: (item.road_id, item.section_id, item.lane_id, item.s))
        route.append(waypoint.transform.location)
    if not route:
        raise RuntimeError("No route available from the configured spawn")
    return route


def record(config, output):
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    if client.get_client_version() != "0.9.16" or client.get_server_version() != "0.9.16":
        raise RuntimeError("Matching CARLA 0.9.16 client and server required")
    world = client.get_world()
    original_settings = world.get_settings()
    if original_settings.synchronous_mode:
        raise RuntimeError("Use a dedicated asynchronous server; another client may own its ticks")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("left", "right", "ground_truth"):
        (output / directory).mkdir()
    actors = []
    traffic_manager = None
    original_weather = None
    metadata = {"status": "incomplete", "schema_version": 1, "config": config}
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        print(f"Preparing {config['map']}", flush=True)
        if world.get_map().name.split("/")[-1] != config["map"]:
            world = client.load_world(config["map"])
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config["fixed_delta_seconds"]
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        print("Reloading world with synchronous settings", flush=True)
        world = client.reload_world(False)
        print("Spawning ego and stereo cameras", flush=True)
        original_weather = world.get_weather()
        world.set_weather(carla.WeatherParameters.ClearNoon)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(config["seed"])
        blueprints = world.get_blueprint_library()
        spawn = world.get_map().get_spawn_points()[config["spawn_index"]]
        vehicle = world.spawn_actor(blueprints.find(config["vehicle"]), spawn)
        actors.append(vehicle)
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        route = make_route(world.get_map(), spawn.location, config)
        traffic_manager.auto_lane_change(vehicle, False)
        traffic_manager.set_desired_speed(vehicle, config["target_speed_kmh"])
        # No other traffic is spawned; avoid a red light making the short test stationary.
        traffic_manager.ignore_lights_percentage(vehicle, 100.0)
        traffic_manager.ignore_signs_percentage(vehicle, 100.0)

        queues = {}
        attributes = {
            "image_size_x": str(config["width"]), "image_size_y": str(config["height"]),
            "fov": str(config["fov"]), "sensor_tick": "0.0",
            "lens_k": "0.0", "lens_kcube": "0.0", "lens_circle_multiplier": "0.0",
            "enable_postprocess_effects": "true", "motion_blur_intensity": "0.0",
            "bloom_intensity": "0.0", "lens_flare_intensity": "0.0",
            "chromatic_aberration_intensity": "0.0",
        }
        for side, sign in (("left", -1), ("right", 1)):
            blueprint = blueprints.find("sensor.camera.rgb")
            for key, value in attributes.items():
                blueprint.set_attribute(key, value)
            mount = carla.Transform(carla.Location(
                x=config["mount_x"], y=sign * config["baseline_m"] / 2, z=config["mount_z"],
            ))
            camera = world.spawn_actor(blueprint, mount, attach_to=vehicle)
            actors.append(camera)
            queues[side] = queue.Queue()
            camera.listen(queues[side].put)
        (output / "calibration.json").write_text(json.dumps(calibration(config), indent=2) + "\n", encoding="utf-8")

        print(f"Warming up {config['warmup_frames']} frames", flush=True)
        for _ in range(config["warmup_frames"]):
            frame = world.tick()
            for sensor_queue in queues.values():
                wait_for_frame(sensor_queue, frame)
        vehicle.set_autopilot(True, 8000)
        traffic_manager.set_path(vehicle, route)
        metadata.update({
            "carla_version": client.get_server_version(), "python_version": platform.python_version(),
            "map": world.get_map().name, "weather": "ClearNoon", "camera_attributes": attributes,
            "spawn_transform_carla": spawn.get_matrix(),
            "requested_route_carla_xyz_m": [[point.x, point.y, point.z] for point in route],
            "traffic_manager": {"port": 8000, "seed": config["seed"], "autopilot": True,
                                "lane_changes": False, "ignore_lights_percent": 100, "ignore_signs_percent": 100},
            "timing": "perf_counter wall time; warmup excluded; image writing includes CSV serialization",
            "png_compress_level": 1,
        })
        with ExitStack() as stack:
            def writer(name, header):
                stream = stack.enter_context((output / name).open("w", newline="", encoding="utf-8"))
                result = csv.writer(stream)
                result.writerow(header)
                return result

            stereo = writer("stereo.csv", ["frame", "timestamp_seconds", "left", "right", "left_frame", "right_frame", "left_timestamp_seconds", "right_timestamp_seconds"])
            poses = {side: writer(f"ground_truth/{side}.csv", ["frame", "timestamp_seconds"] +
                                  [f"t{r}{c}" for r in range(4) for c in range(4)])
                     for side in ("left", "right", "vehicle")}
            controls = writer("controls.csv", ["frame", "throttle", "steer", "brake", "hand_brake", "reverse", "gear"])
            timings = writer("timings.csv", ["frame", "tick_ms", "sensor_wait_ms", "write_ms", "total_ms"])
            started = time.perf_counter()
            for index in range(config["frames"]):
                begin = time.perf_counter()
                frame = world.tick()
                ticked = time.perf_counter()
                pair = {side: wait_for_frame(sensor_queue, frame) for side, sensor_queue in queues.items()}
                snapshot = world.get_snapshot()
                timestamp = snapshot.timestamp.elapsed_seconds
                if snapshot.frame != frame or any(abs(image.timestamp - timestamp) > 1e-5 for image in pair.values()):
                    raise RuntimeError("World and stereo measurements do not share a frame and timestamp")
                received = time.perf_counter()
                name = f"{frame:08d}.png"
                for side, image in pair.items():
                    save_rgb(image, output / side / name)
                    poses[side].writerow([frame, f"{image.timestamp:.9f}"] + optical_pose(image.transform.get_matrix()).ravel().tolist())
                vehicle_transform = snapshot.find(vehicle.id).get_transform()
                poses["vehicle"].writerow([frame, f"{timestamp:.9f}"] + vehicle_pose(vehicle_transform.get_matrix()).ravel().tolist())
                stereo.writerow([frame, f"{timestamp:.9f}", f"left/{name}", f"right/{name}",
                                 pair["left"].frame, pair["right"].frame,
                                 f"{pair['left'].timestamp:.9f}", f"{pair['right'].timestamp:.9f}"])
                control = vehicle.get_control()
                controls.writerow([frame, control.throttle, control.steer, control.brake, control.hand_brake, control.reverse, control.gear])
                written = time.perf_counter()
                timings.writerow([frame, (ticked - begin) * 1000, (received - ticked) * 1000,
                                  (written - received) * 1000, (written - begin) * 1000])
                if (index + 1) % 100 == 0:
                    print(f"Recorded {index + 1}/{config['frames']} stereo pairs", flush=True)
            metadata["capture_wall_seconds"] = time.perf_counter() - started
    finally:
        if actors and actors[0].type_id.startswith("vehicle."):
            actors[0].set_autopilot(False, 8000)
        for actor in reversed(actors):
            if actor.type_id.startswith("sensor."):
                actor.stop()
            actor.destroy()
        if traffic_manager is not None:
            traffic_manager.set_synchronous_mode(False)
        if original_weather is not None:
            world.set_weather(original_weather)
        world.apply_settings(original_settings)
    metadata["status"] = "complete"
    metadata["cleanup_status"] = "complete"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Captured {config['frames']} stereo pairs in {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stereo.json")
    parser.add_argument("--frames", type=int, help="Override frame count for a shorter smoke run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.frames is not None:
        config["frames"] = args.frames
    if config["frames"] <= 0:
        parser.error("frames must be positive")
    output = args.output or ROOT / "runs" / datetime.now().strftime("stereo_%Y%m%d_%H%M%S_%f")
    record(config, output.resolve())


if __name__ == "__main__":
    main()
