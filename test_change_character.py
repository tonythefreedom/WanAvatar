#!/usr/bin/env python3
"""
Test script for Change Character V1.1 workflow via ComfyUI.

Usage:
  python test_change_character.py

This script:
1. Checks ComfyUI is running (starts it if needed)
2. Uploads image and video to ComfyUI
3. Prepares and submits the Change Character V1.1 workflow
4. Monitors execution via WebSocket
5. Retrieves and saves the output video
"""

import json
import os
import subprocess
import sys
import time
import uuid
import requests
import websocket as ws_client

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_WS_URL = "ws://127.0.0.1:8188/ws"

# Test files
IMAGE_PATH = "uploads/lina_stage.jpg"
VIDEO_PATH = "uploads/dance_ref_1080p.mp4"
WORKFLOW_PATH = "workflow/Change_character_V1.1_api.json"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_comfyui():
    """Check if ComfyUI is running."""
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
        if resp.status_code == 200:
            print("[OK] ComfyUI is running")
            return True
    except Exception:
        pass
    print("[ERROR] ComfyUI is not running at", COMFYUI_URL)
    return False


def upload_file(local_path: str, subfolder: str = "") -> str:
    """Upload a file to ComfyUI's input directory."""
    filename = os.path.basename(local_path)
    print(f"  Uploading {filename}...")
    with open(local_path, 'rb') as f:
        resp = requests.post(
            f"{COMFYUI_URL}/upload/image",
            files={"image": (filename, f, "application/octet-stream")},
            data={"overwrite": "true", "subfolder": subfolder},
        )
    if resp.status_code != 200:
        print(f"  [ERROR] Upload failed: {resp.text}")
        raise Exception(f"Upload failed: {resp.text}")
    result_name = resp.json().get("name", filename)
    print(f"  [OK] Uploaded as: {result_name}")
    return result_name


def resolve_set_get_nodes(wf: dict) -> dict:
    """
    Resolve SetNode/GetNode virtual nodes into direct connections.

    SetNode/GetNode are frontend-only JavaScript nodes in KJNodes that have
    NO Python backend. They cannot be executed via ComfyUI's API. This function
    rewires all connections to bypass them.

    Also removes Note nodes and Fast Groups Bypasser nodes.
    """
    # Phase 1: Build SetNode maps
    # name_to_source: "Variable Name" -> [source_node_id, source_output_index]
    # setnode_ids: set of SetNode node IDs
    name_to_source = {}
    setnode_ids = set()
    getnode_ids = set()
    remove_ids = set()

    for node_id, node in wf.items():
        class_type = node.get("class_type", "")

        if class_type == "SetNode":
            setnode_ids.add(node_id)
            remove_ids.add(node_id)
            name = node["inputs"].get("value", "")
            # Find the source connection (the non-"value" input)
            for key, val in node["inputs"].items():
                if key != "value" and isinstance(val, list) and len(val) == 2:
                    name_to_source[name] = val  # [source_node_id, output_index]
                    break
            print(f"  SetNode {node_id}: '{name}' -> {name_to_source.get(name, '?')}")

        elif class_type == "GetNode":
            getnode_ids.add(node_id)
            remove_ids.add(node_id)

        elif class_type in ("Note", "Fast Groups Bypasser (rgthree)"):
            remove_ids.add(node_id)

    # Phase 2: Build GetNode resolution map
    # getnode_id -> [source_node_id, source_output_index]
    getnode_to_source = {}
    for node_id in getnode_ids:
        name = wf[node_id]["inputs"].get("value", "")
        if name in name_to_source:
            getnode_to_source[node_id] = name_to_source[name]
            print(f"  GetNode {node_id}: '{name}' -> {getnode_to_source[node_id]}")
        else:
            print(f"  [WARN] GetNode {node_id}: '{name}' has no matching SetNode!")

    # Phase 3: Build SetNode output resolution map
    # When a node references a SetNode output directly (e.g., ["49", 0]),
    # it should resolve to the SetNode's source
    setnode_to_source = {}
    for node_id in setnode_ids:
        name = wf[node_id]["inputs"].get("value", "")
        if name in name_to_source:
            setnode_to_source[node_id] = name_to_source[name]

    # Phase 4: Rewire all references
    rewire_count = 0
    for node_id, node in wf.items():
        if node_id in remove_ids:
            continue
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2:
                ref_id = str(val[0])
                # If referencing a GetNode, replace with the GetNode's source
                if ref_id in getnode_to_source:
                    old_ref = val.copy()
                    node["inputs"][key] = getnode_to_source[ref_id]
                    rewire_count += 1
                    print(f"  Rewired {node_id}.{key}: GetNode[{ref_id}] -> {node['inputs'][key]}")
                # If referencing a SetNode output, replace with SetNode's source
                elif ref_id in setnode_to_source:
                    old_ref = val.copy()
                    node["inputs"][key] = setnode_to_source[ref_id]
                    rewire_count += 1
                    print(f"  Rewired {node_id}.{key}: SetNode[{ref_id}] -> {node['inputs'][key]}")

    # Phase 5: Fix class_type mismatches (display name vs registered name)
    CLASS_TYPE_FIXES = {
        "Int": "easy int",
    }
    for node_id, node in wf.items():
        if node_id in remove_ids:
            continue
        ct = node.get("class_type", "")
        if ct in CLASS_TYPE_FIXES:
            node["class_type"] = CLASS_TYPE_FIXES[ct]
            print(f"  Fixed class_type: node {node_id} '{ct}' -> '{CLASS_TYPE_FIXES[ct]}'")

    # Phase 6: Remove virtual nodes
    for node_id in remove_ids:
        del wf[node_id]

    print(f"\n  Resolved: {len(setnode_ids)} SetNodes, {len(getnode_ids)} GetNodes, "
          f"{len(remove_ids)} total removed, {rewire_count} connections rewired")

    return wf


def prepare_workflow(image_name: str, video_name: str, prompt: str, width: int, height: int) -> dict:
    """Load and prepare the workflow JSON with user parameters."""
    with open(WORKFLOW_PATH) as f:
        wf = json.load(f)

    # Step 1: Resolve SetNode/GetNode virtual nodes
    print("  Resolving SetNode/GetNode virtual nodes...")
    wf = resolve_set_get_nodes(wf)

    # Step 2: Set user parameters
    # Set image in node 91 (LoadImage)
    if "91" in wf:
        wf["91"]["inputs"]["image"] = image_name
        print(f"  Node 91 (LoadImage): {image_name}")

    # Set video in node 114 (VHS_LoadVideo)
    if "114" in wf:
        wf["114"]["inputs"]["video"] = video_name
        print(f"  Node 114 (VHS_LoadVideo): {video_name}")

    # Set prompt in node 209
    if "209" in wf:
        wf["209"]["inputs"]["positive_prompt"] = prompt
        print(f"  Node 209 (Prompt): {prompt[:50]}...")

    # Set resolution (width in node 123, height in node 124)
    if "123" in wf:
        wf["123"]["inputs"]["value"] = width
        print(f"  Node 123 (Width): {width}")
    if "124" in wf:
        wf["124"]["inputs"]["value"] = height
        print(f"  Node 124 (Height): {height}")

    return wf


def submit_workflow(workflow: dict, client_id: str) -> str:
    """Submit workflow to ComfyUI."""
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
    )
    if resp.status_code != 200:
        error_text = resp.text
        print(f"[ERROR] ComfyUI rejected workflow: {error_text}")
        # Try to parse error for missing nodes
        try:
            error_data = resp.json()
            if "node_errors" in error_data:
                for node_id, err in error_data["node_errors"].items():
                    print(f"  Node {node_id}: {err}")
            if "error" in error_data:
                err_info = error_data["error"]
                print(f"  Error type: {err_info.get('type', 'unknown')}")
                print(f"  Message: {err_info.get('message', 'no message')}")
        except Exception:
            pass
        raise Exception(f"Workflow rejected: {error_text}")

    prompt_id = resp.json()["prompt_id"]
    print(f"[OK] Workflow submitted: prompt_id={prompt_id}")
    return prompt_id


def monitor_progress(prompt_id: str, client_id: str):
    """Monitor execution via WebSocket."""
    ws = ws_client.WebSocket()
    ws.settimeout(600)
    ws.connect(f"{COMFYUI_WS_URL}?clientId={client_id}")

    completed = 0
    total_nodes = "?"
    start_time = time.time()

    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                msg_type = data.get("type")
                msg_data = data.get("data", {})

                if msg_type == "progress":
                    value = msg_data.get("value", 0)
                    max_val = msg_data.get("max", 1)
                    elapsed = time.time() - start_time
                    print(f"\r  [{elapsed:.0f}s] Node {completed}/{total_nodes} - step {value}/{max_val}   ", end="", flush=True)

                elif msg_type == "executing":
                    node_id = msg_data.get("node")
                    if node_id is None and msg_data.get("prompt_id") == prompt_id:
                        elapsed = time.time() - start_time
                        print(f"\n[OK] Execution complete in {elapsed:.1f}s")
                        break
                    completed += 1

                elif msg_type == "execution_error":
                    error = msg_data.get("exception_message", "Unknown")
                    node_id = msg_data.get("node_id", "?")
                    node_type = msg_data.get("node_type", "?")
                    traceback_str = "\n".join(msg_data.get("traceback", []))
                    print(f"\n[ERROR] ComfyUI execution error at node {node_id} ({node_type}):")
                    print(f"  {error}")
                    if traceback_str:
                        print(f"  Traceback:\n{traceback_str}")
                    raise Exception(f"Execution error: {error}")

                elif msg_type == "execution_cached":
                    cached_nodes = msg_data.get("nodes", [])
                    if cached_nodes:
                        print(f"  Cached {len(cached_nodes)} nodes")

                elif msg_type == "status":
                    queue = msg_data.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                    if queue > 0:
                        print(f"  Queue remaining: {queue}")

    finally:
        ws.close()


def retrieve_output(prompt_id: str) -> str:
    """Retrieve output video from ComfyUI history."""
    resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    history = resp.json()

    outputs = history.get(prompt_id, {}).get("outputs", {})

    for node_id, node_output in outputs.items():
        # Check for video output (VHS_VideoCombine → 'gifs' key)
        if "gifs" in node_output:
            for gif in node_output["gifs"]:
                filename = gif["filename"]
                subfolder = gif.get("subfolder", "")
                file_type = gif.get("type", "temp")

                view_resp = requests.get(
                    f"{COMFYUI_URL}/view",
                    params={"filename": filename, "subfolder": subfolder, "type": file_type},
                )
                if view_resp.status_code != 200:
                    continue

                output_filename = f"test_change_char_{int(time.time())}.mp4"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, 'wb') as f:
                    f.write(view_resp.content)
                return output_path

        # Check for image output
        if "images" in node_output:
            for img in node_output["images"]:
                filename = img["filename"]
                print(f"  Image output: {filename} (node {node_id})")

    raise Exception("No output video found in ComfyUI history")


def main():
    print("=" * 60)
    print("Change Character V1.1 - ComfyUI Workflow Test")
    print("=" * 60)

    # Step 1: Check ComfyUI
    print("\n[Step 1] Checking ComfyUI...")
    if not check_comfyui():
        print("Please start ComfyUI first.")
        sys.exit(1)

    # Step 2: Upload files
    print("\n[Step 2] Uploading files to ComfyUI...")
    image_name = upload_file(IMAGE_PATH)
    video_name = upload_file(VIDEO_PATH)

    # Step 3: Prepare workflow
    print("\n[Step 3] Preparing workflow...")
    workflow = prepare_workflow(
        image_name=image_name,
        video_name=video_name,
        prompt="A beautiful Korean idol dancing smoothly on stage, detailed face, flowing hair, dynamic lighting, high quality",
        width=576,
        height=1024,
    )

    # Step 4: Submit
    print("\n[Step 4] Submitting workflow...")
    client_id = str(uuid.uuid4())
    prompt_id = submit_workflow(workflow, client_id)

    # Step 5: Monitor
    print("\n[Step 5] Monitoring execution...")
    monitor_progress(prompt_id, client_id)

    # Step 6: Retrieve output
    print("\n[Step 6] Retrieving output...")
    output_path = retrieve_output(prompt_id)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output video: {output_path} ({size_mb:.1f} MB)")

    # Step 7: Merge audio from reference video
    print("\n[Step 7] Merging audio from reference video...")
    merged_path = output_path.replace(".mp4", "_audio.mp4")
    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", output_path,
            "-i", VIDEO_PATH,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            merged_path,
        ], capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(merged_path) and os.path.getsize(merged_path) > 0:
            os.replace(merged_path, output_path)
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[OK] Audio merged: {output_path} ({size_mb:.1f} MB)")
        else:
            print(f"[WARN] ffmpeg failed: {result.stderr.decode()[:300]}")
    except Exception as e:
        print(f"[WARN] Audio merge error: {e}")

    print(f"\n[SUCCESS] Final output: {output_path}")


if __name__ == "__main__":
    main()
