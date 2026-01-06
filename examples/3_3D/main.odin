
package main

import log "core:log"
import "../../gpu"
import "core:math"
import "core:math/linalg"
import "gltf2"
import "base:runtime"
import "core:fmt"
import "core:os"

import sdl "vendor:sdl3"

Window_Size_X :: 1000
Window_Size_Y :: 1000
Frames_In_Flight :: 3
Example_Name :: "3D"

Sponza_Scene :: #load("assets/sponza.glb")

main :: proc()
{
    fmt.println("Work in progress! (Needs depth buffer)")

    ok_i := sdl.Init({ .VIDEO })
    assert(ok_i)

    console_logger := log.create_console_logger()
    defer log.destroy_console_logger(console_logger)
    context.logger = console_logger

    ts_freq := sdl.GetPerformanceFrequency()
    max_delta_time: f32 = 1.0 / 10.0  // 10fps

    window_flags :: sdl.WindowFlags {
        .HIGH_PIXEL_DENSITY,
        .VULKAN,
    }
    window := sdl.CreateWindow(Example_Name, Window_Size_X, Window_Size_Y, window_flags)
    ensure(window != nil)

    gpu.init(window, Frames_In_Flight)
    defer gpu.cleanup()

    vert_shader := gpu.shader_create(#load("shaders/test.vert.spv", []u32), .Vertex)
    frag_shader := gpu.shader_create(#load("shaders/test.frag.spv", []u32), .Fragment)
    defer {
        gpu.shader_destroy(&vert_shader)
        gpu.shader_destroy(&frag_shader)
    }

    Vertex :: struct { pos: [4]f32, color: [4]f32 }

    upload_arena := gpu.arena_init(1024 * 1024 * 1024)
    defer gpu.arena_destroy(&upload_arena)

    queue := gpu.get_queue()

    upload_cmd_buf := gpu.commands_begin(queue)
    scene := load_scene_gltf(Sponza_Scene, &upload_arena, upload_cmd_buf)
    defer destroy_scene(&scene)
    gpu.cmd_barrier(upload_cmd_buf, .Transfer, .All, {})
    gpu.queue_submit(queue, { upload_cmd_buf })

    now_ts := sdl.GetPerformanceCounter()

    frame_arenas: [Frames_In_Flight]gpu.Arena
    for &frame_arena in frame_arenas do frame_arena = gpu.arena_init(10 * 1024 * 1024)
    defer for &frame_arena in frame_arenas do gpu.arena_destroy(&frame_arena)
    next_frame := u64(1)
    frame_sem := gpu.semaphore_create(0)
    defer gpu.semaphore_destroy(&frame_sem)
    for true
    {
        proceed := handle_window_events(window)
        if !proceed do break
        if .MINIMIZED in sdl.GetWindowFlags(window)
        {
            sdl.Delay(16)
            continue
        }

        if next_frame > Frames_In_Flight {
            gpu.semaphore_wait(frame_sem, next_frame - Frames_In_Flight)
        }

        last_ts := now_ts
        now_ts = sdl.GetPerformanceCounter()
        delta_time := min(max_delta_time, f32(f64((now_ts - last_ts)*1000) / f64(ts_freq)) / 1000.0)

        world_to_view := first_person_camera_view(delta_time)
        aspect_ratio := f32(Window_Size_X) / f32(Window_Size_Y)
        view_to_proj := linalg.matrix4_perspective_f32(math.RAD_PER_DEG * 59.0, aspect_ratio, 0.1, 1000.0, false)

        frame_arena := &frame_arenas[next_frame % Frames_In_Flight]

        swapchain := gpu.swapchain_acquire_next()  // Blocks CPU until at least one frame is available.

        cmd_buf := gpu.commands_begin(queue)
        gpu.cmd_begin_render_pass(cmd_buf, {
            color_attachments = {
                { texture = swapchain, clear_color = { 0.7, 0.7, 0.7, 1.0 } }
            }
        })
        gpu.cmd_set_shaders(cmd_buf, vert_shader, frag_shader)

        for instance in scene.instances
        {
            mesh := scene.meshes[instance.mesh_idx]

            Vert_Data :: struct #all_or_none {
                model_to_world: matrix[4, 4]f32,
                model_to_world_normal: matrix[4, 4]f32,
                world_to_view: matrix[4, 4]f32,
                view_to_proj: matrix[4, 4]f32,
                positions: rawptr,
                normals: rawptr,
            }
            verts_data := gpu.arena_alloc(frame_arena, Vert_Data)
            verts_data.cpu^ = {
                positions = mesh.pos,
                normals = mesh.normals,
                model_to_world = instance.transform,
                model_to_world_normal = linalg.transpose(linalg.inverse(instance.transform)),
                world_to_view = world_to_view,
                view_to_proj = view_to_proj,
            }

            gpu.cmd_draw_indexed_instanced(cmd_buf, verts_data.gpu, nil, mesh.indices, mesh.idx_count, 1)
        }

        gpu.cmd_end_render_pass(cmd_buf)
        gpu.queue_submit(queue, { cmd_buf }, frame_sem, next_frame)

        gpu.swapchain_present(queue, frame_sem, next_frame)
        next_frame += 1

        gpu.arena_free_all(frame_arena)
    }

    gpu.wait_idle()
}

Mesh :: struct
{
    pos: rawptr,
    normals: rawptr,
    indices: rawptr,
    idx_count: u32,
}

upload_mesh :: proc(upload_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer, positions: [][4]f32, normals: [][4]f32, indices: []u32) -> Mesh
{
    assert(len(positions) == len(normals))

    positions_staging := gpu.arena_alloc_array(upload_arena, [4]f32, len(positions))
    normals_staging := gpu.arena_alloc_array(upload_arena, [4]f32, len(normals))
    indices_staging := gpu.arena_alloc_array(upload_arena, u32, len(indices))
    copy(positions_staging.cpu, positions)
    copy(normals_staging.cpu, normals)
    copy(indices_staging.cpu, indices)

    res: Mesh
    res.pos = gpu.mem_alloc_typed_gpu([4]f32, len(positions))
    res.normals = gpu.mem_alloc_typed_gpu([4]f32, len(normals))
    res.indices = gpu.mem_alloc_typed_gpu(u32, len(indices))
    res.idx_count = u32(len(indices))
    gpu.cmd_mem_copy(cmd_buf, positions_staging.gpu, res.pos, u64(len(positions) * size_of(positions[0])))
    gpu.cmd_mem_copy(cmd_buf, normals_staging.gpu, res.normals, u64(len(normals) * size_of(normals[0])))
    gpu.cmd_mem_copy(cmd_buf, indices_staging.gpu, res.indices, u64(len(indices) * size_of(indices[0])))
    return res
}

destroy_mesh :: proc(mesh: ^Mesh)
{
    gpu.mem_free(mesh.pos)
    gpu.mem_free(mesh.normals)
    gpu.mem_free(mesh.indices)
    mesh^ = {}
}

/////////////////////////////////////////
// Miscellaneous (Can be ignored)

Scene :: struct
{
    meshes: [dynamic]Mesh,
    instances: [dynamic]Instance,
}

destroy_scene :: proc(scene: ^Scene)
{
    for &mesh in scene.meshes {
        destroy_mesh(&mesh)
    }

    delete(scene.meshes)
    delete(scene.instances)
    scene^ = {}
}

Instance :: struct
{
    transform: matrix[4, 4]f32,
    mesh_idx: u32,
}

// Input

Key_State :: struct
{
    pressed: bool,
    pressing: bool,
    released: bool,
}

Input :: struct
{
    pressing_right_click: bool,
    keys: #sparse[sdl.Scancode]Key_State,

    mouse_dx: f32,  // pixels/dpi (inches), right is positive
    mouse_dy: f32,  // pixels/dpi (inches), up is positive
}

INPUT: Input

handle_window_events :: proc(window: ^sdl.Window) -> (proceed: bool)
{
    // Reset "one-shot" inputs
    for &key in INPUT.keys
    {
        key.pressed = false
        key.released = false
    }
    INPUT.mouse_dx = 0
    INPUT.mouse_dy = 0

    event: sdl.Event
    proceed = true
    for sdl.PollEvent(&event)
    {
        #partial switch event.type
        {
            case .QUIT:
                proceed = false
            case .WINDOW_CLOSE_REQUESTED:
            {
                if event.window.windowID == sdl.GetWindowID(window) {
                    proceed = false
                }
            }
            // Input events
            case .MOUSE_BUTTON_DOWN, .MOUSE_BUTTON_UP:
            {
                event := event.button
                if event.type == .MOUSE_BUTTON_DOWN {
                    if event.button == sdl.BUTTON_RIGHT {
                        INPUT.pressing_right_click = true
                    }
                } else if event.type == .MOUSE_BUTTON_UP {
                    if event.button == sdl.BUTTON_RIGHT {
                        INPUT.pressing_right_click = false
                    }
                }
            }
            case .KEY_DOWN, .KEY_UP:
            {
                event := event.key
                if event.repeat do break

                if event.type == .KEY_DOWN
                {
                    INPUT.keys[event.scancode].pressed = true
                    INPUT.keys[event.scancode].pressing = true
                }
                else
                {
                    INPUT.keys[event.scancode].pressing = false
                    INPUT.keys[event.scancode].released = true
                }
            }
            case .MOUSE_MOTION:
            {
                event := event.motion
                INPUT.mouse_dx += event.xrel
                INPUT.mouse_dy -= event.yrel  // In sdl, up is negative
            }
        }
    }

    return
}

first_person_camera_view :: proc(delta_time: f32) -> matrix[4, 4]f32
{
    @(static) cam_pos: [3]f32 = { 0, 2.5, -10 }

    @(static) angle: [2]f32

    cam_rot: quaternion128 = 1

    mouse_sensitivity := math.to_radians_f32(0.2)  // Radians per pixel
    mouse: [2]f32
    if INPUT.pressing_right_click
    {
        mouse.x = INPUT.mouse_dx * mouse_sensitivity
        mouse.y = INPUT.mouse_dy * mouse_sensitivity
    }

    angle += mouse

    // Wrap angle.x
    for angle.x < 0 do angle.x += 2*math.PI
    for angle.x > 2*math.PI do angle.x -= 2*math.PI

    angle.y = clamp(angle.y, math.to_radians_f32(-90), math.to_radians_f32(90))
    y_rot := linalg.quaternion_angle_axis(angle.y, [3]f32 { -1, 0, 0 })
    x_rot := linalg.quaternion_angle_axis(angle.x, [3]f32 { 0, 1, 0 })
    cam_rot = x_rot * y_rot

    // Movement
    @(static) cur_vel: [3]f32
    move_speed: f32 : 6.0
    move_speed_fast: f32 : 15.0
    move_accel: f32 : 300.0

    keyboard_dir_xz: [3]f32
    keyboard_dir_y: f32
    if INPUT.pressing_right_click
    {
        keyboard_dir_xz.x = f32(int(INPUT.keys[.D].pressing) - int(INPUT.keys[.A].pressing))
        keyboard_dir_xz.z = f32(int(INPUT.keys[.W].pressing) - int(INPUT.keys[.S].pressing))
        keyboard_dir_y    = f32(int(INPUT.keys[.E].pressing) - int(INPUT.keys[.Q].pressing))

        // It's a "direction" input so its length
        // should be no more than 1
        if linalg.dot(keyboard_dir_xz, keyboard_dir_xz) > 1 {
            keyboard_dir_xz = linalg.normalize(keyboard_dir_xz)
        }

        if abs(keyboard_dir_y) > 1 {
            keyboard_dir_y = math.sign(keyboard_dir_y)
        }
    }

    target_vel := keyboard_dir_xz * move_speed
    target_vel = linalg.quaternion_mul_vector3(cam_rot, target_vel)
    target_vel.y += keyboard_dir_y * move_speed

    cur_vel = approach_linear(cur_vel, target_vel, move_accel * delta_time)
    cam_pos += cur_vel * delta_time

    return world_to_view_mat(cam_pos, cam_rot)

    approach_linear :: proc(cur: [3]f32, target: [3]f32, delta: f32) -> [3]f32
    {
        diff := target - cur
        dist := linalg.length(diff)

        if dist <= delta do return target
        return cur + diff / dist * delta
    }
}

world_to_view_mat :: proc(cam_pos: [3]f32, cam_rot: quaternion128) -> matrix[4, 4]f32
{
    view_rot := linalg.normalize(linalg.quaternion_inverse(cam_rot))
    view_pos := -cam_pos
    return #force_inline linalg.matrix4_from_quaternion(view_rot) *
           #force_inline linalg.matrix4_translate(view_pos)
}

// I/O

load_scene_gltf :: proc(contents: []byte, upload_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer) -> Scene
{
    options := gltf2.Options {}
    options.is_glb = true
    data, err_l := gltf2.parse(contents, options)
    switch err in err_l
    {
        case gltf2.JSON_Error: log.error(err)
        case gltf2.GLTF_Error: log.error(err)
    }
    defer gltf2.unload(data)

    // Load meshes
    meshes: [dynamic]Mesh
    start_idx: [dynamic]u32
    defer delete(start_idx)
    for mesh, i in data.meshes
    {
        append(&start_idx, u32(len(meshes)))

        for primitive, j in mesh.primitives
        {
            // ???
            buffer_view := &data.buffer_views[data.accessors[primitive.attributes["POSITION"]].buffer_view.?]
            assert(buffer_view.byte_stride == 12 || buffer_view.byte_stride == nil)
            buffer_view.byte_stride = nil
            buffer_view = &data.buffer_views[data.accessors[primitive.attributes["NORMAL"]].buffer_view.?]
            assert(buffer_view.byte_stride == 12 || buffer_view.byte_stride == nil)
            buffer_view.byte_stride = nil
            buffer_view = &data.buffer_views[data.accessors[primitive.attributes["TEXCOORD_0"]].buffer_view.?]
            assert(buffer_view.byte_stride == 8 || buffer_view.byte_stride == nil)
            buffer_view.byte_stride = nil

            assert(primitive.mode == .Triangles)

            positions := gltf2.buffer_slice(data, primitive.attributes["POSITION"])
            normals := gltf2.buffer_slice(data, primitive.attributes["NORMAL"])
            uvs := gltf2.buffer_slice(data, primitive.attributes["TEXCOORD_0"])
            lm_uvs := gltf2.buffer_slice(data, primitive.attributes["TEXCOORD_1"])
            indices := gltf2.buffer_slice(data, primitive.indices.?)

            indices_u32: [dynamic]u32
            defer delete(indices_u32)
            #partial switch ids in indices
            {
                case []u16:
                    for i in 0..<len(ids) do append(&indices_u32, u32(ids[i]))
                case []u32:
                    for i in 0..<len(ids) do append(&indices_u32, ids[i])
                case: assert(false)
            }

            pos_final := to_vec4_array(positions.([][3]f32), allocator = context.temp_allocator)
            normals_final := to_vec4_array(normals.([][3]f32), allocator = context.temp_allocator)
            loaded := upload_mesh(upload_arena, cmd_buf, pos_final, normals_final, indices_u32[:])
            append(&meshes, loaded)
        }
    }

    // Load instances
    instances: [dynamic]Instance
    for node_idx in data.scenes[0].nodes
    {
        node := data.nodes[node_idx]

        traverse_node(&instances, data, 1, int(node_idx), meshes, start_idx)

        traverse_node :: proc(instances: ^[dynamic]Instance, data: ^gltf2.Data, parent_transform: matrix[4, 4]f32, node_idx: int, meshes: [dynamic]Mesh, start_idx: [dynamic]u32)
        {
            node := data.nodes[node_idx]

            flip_z: matrix[4, 4]f32 = 1
            flip_z[2, 2] = -1
            local_transform := xform_to_mat(node.translation, node.rotation, node.scale)
            transform := parent_transform * local_transform
            if node.mesh != nil
            {
                mesh_idx := node.mesh.?
                mesh := data.meshes[mesh_idx]

                for primitive, j in mesh.primitives
                {
                    primitive_idx := start_idx[mesh_idx] + u32(j)
                    instance := Instance {
                        transform = flip_z * transform,
                        mesh_idx = primitive_idx,
                    }
                    append(instances, instance)
                }
            }

            for child in node.children {
                traverse_node(instances, data, transform, int(child), meshes, start_idx)
            }
        }
    }

    return {
        instances = instances,
        meshes = meshes,
    }
}

to_vec4_array :: proc(array: [][3]f32, allocator: runtime.Allocator) -> [][4]f32
{
    res := make([][4]f32, len(array), allocator = allocator)
    for &v, i in res do v = { array[i].x, array[i].y, array[i].z, 0.0 }
    return res
}

xform_to_mat_f64 :: proc(pos: [3]f64, rot: quaternion256, scale: [3]f64) -> matrix[4, 4]f32
{
    return cast(matrix[4, 4]f32) (#force_inline linalg.matrix4_translate(pos) *
           #force_inline linalg.matrix4_from_quaternion(rot) *
           #force_inline linalg.matrix4_scale(scale))
}

xform_to_mat_f32 :: proc(pos: [3]f32, rot: quaternion128, scale: [3]f32) -> matrix[4, 4]f32
{
    return #force_inline linalg.matrix4_translate(pos) *
           #force_inline linalg.matrix4_from_quaternion(rot) *
           #force_inline linalg.matrix4_scale(scale)
}

xform_to_mat :: proc {
    xform_to_mat_f32,
    xform_to_mat_f64,
}
