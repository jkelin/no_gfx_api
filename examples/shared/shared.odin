
package shared

import log "core:log"
import "../../gpu"
import "core:math"
import "core:math/linalg"
import "gltf2"
import "base:runtime"
import "core:image"
import "core:image/png"
import "core:image/jpeg"
import "core:os"
import "core:fmt"
import "core:path/filepath"

import sdl "vendor:sdl3"

Mesh :: struct
{
    pos: rawptr,
    normals: rawptr,
    uvs: rawptr,
    indices: rawptr,
    idx_count: u32,
    albedo_texture_id: Maybe(u32),
    normal_texture_id: Maybe(u32),
}

upload_mesh :: proc(upload_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer, positions: [][4]f32, normals: [][4]f32, uvs: [][2]f32, indices: []u32) -> Mesh
{
    assert(len(positions) == len(normals))
    assert(len(positions) == len(uvs))

    positions_staging := gpu.arena_alloc_array(upload_arena, [4]f32, len(positions))
    normals_staging := gpu.arena_alloc_array(upload_arena, [4]f32, len(normals))
    uvs_staging := gpu.arena_alloc_array(upload_arena, [2]f32, len(uvs))
    indices_staging := gpu.arena_alloc_array(upload_arena, u32, len(indices))
    copy(positions_staging.cpu, positions)
    copy(normals_staging.cpu, normals)
    copy(uvs_staging.cpu, uvs)
    copy(indices_staging.cpu, indices)

    res: Mesh
    res.pos = gpu.mem_alloc_typed_gpu([4]f32, len(positions))
    res.normals = gpu.mem_alloc_typed_gpu([4]f32, len(normals))
    res.uvs = gpu.mem_alloc_typed_gpu([2]f32, len(uvs))
    res.indices = gpu.mem_alloc_typed_gpu(u32, len(indices))
    res.idx_count = u32(len(indices))
    gpu.cmd_mem_copy(cmd_buf, positions_staging.gpu, res.pos, u64(len(positions) * size_of(positions[0])))
    gpu.cmd_mem_copy(cmd_buf, normals_staging.gpu, res.normals, u64(len(normals) * size_of(normals[0])))
    gpu.cmd_mem_copy(cmd_buf, uvs_staging.gpu, res.uvs, u64(len(uvs) * size_of(uvs[0])))
    gpu.cmd_mem_copy(cmd_buf, indices_staging.gpu, res.indices, u64(len(indices) * size_of(indices[0])))
    return res
}

destroy_mesh :: proc(mesh: ^Mesh)
{
    gpu.mem_free(mesh.pos)
    gpu.mem_free(mesh.normals)
    gpu.mem_free(mesh.uvs)
    gpu.mem_free(mesh.indices)
    mesh^ = {}
}

Scene :: struct
{
    meshes: [dynamic]Mesh,
    instances: [dynamic]Instance,
    textures: [dynamic]gpu.Owned_Texture,
}

destroy_scene :: proc(scene: ^Scene)
{
    for &mesh in scene.meshes {
        destroy_mesh(&mesh)
    }
    
    for &tex in scene.textures {
        gpu.free_and_destroy_texture(&tex)
    }

    delete(scene.meshes)
    delete(scene.instances)
    delete(scene.textures)
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
    pressed_left_click: bool,
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
    INPUT.pressed_left_click = false
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
                    } else if event.button == sdl.BUTTON_LEFT {
                        INPUT.pressed_left_click = true
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
    @(static) cam_pos: [3]f32 = { -7.581631, 1.1906259, 0.25928685 }

    @(static) angle: [2]f32 = { 1.570796, 0.3665192 }

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

load_texture_from_gltf_image :: proc(data: ^gltf2.Data, image_idx: u32, gltf_dir: string, upload_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer) -> (texture: gpu.Owned_Texture, ok: bool)
{
    if int(image_idx) >= len(data.images) {
        return {}, false
    }
    
    img := &data.images[image_idx]
    image_bytes: []byte
    
    // Handle buffer_view (embedded image)
    if img.buffer_view != nil {
        buffer_view := &data.buffer_views[img.buffer_view.?]
        uri := data.buffers[buffer_view.buffer].uri
        #partial switch v in uri {
        case []byte:
            fmt.println("Loading embedded texture:", image_idx)
            start_byte := buffer_view.byte_offset
            image_bytes = v[start_byte:start_byte + buffer_view.byte_length]
        case:
            return {}, false
        }
    } else {
        // Handle URI (file path)
        #partial switch v in img.uri {
        case string:
            full_path := fmt.tprintf("%s/%s", gltf_dir, v)
            fmt.println("Loading texture from:", full_path)
            bytes, ok_file := os.read_entire_file(full_path)
            if !ok_file {
                return {}, false
            }
            image_bytes = bytes
        case []byte:
            fmt.println("Loading texture:", image_idx)
            image_bytes = v
        case:
            return {}, false
        }
    }
    
    // Load image
    options := image.Options {
        .alpha_add_if_missing,
    }
    img_data, err := image.load_from_bytes(image_bytes, options)
    if err != nil {
        return {}, false
    }
    defer image.destroy(img_data)
    
    // Upload to GPU
    staging, staging_gpu := gpu.arena_alloc_untyped(upload_arena, u64(len(img_data.pixels.buf)))
    runtime.mem_copy(staging, raw_data(img_data.pixels.buf), len(img_data.pixels.buf))
    
    texture = gpu.alloc_and_create_texture({
        type = .D2,
        dimensions = { u32(img_data.width), u32(img_data.height), 1 },
        mip_count = 1,
        layer_count = 1,
        sample_count = 1,
        format = .RGBA8_Unorm,
        usage = { .Sampled },
    })
    gpu.cmd_copy_to_texture(cmd_buf, texture, staging_gpu, texture.mem)
    
    return texture, true
}

load_textures_from_gltf :: proc(data: ^gltf2.Data, gltf_dir: string, upload_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer) -> (textures: [dynamic]gpu.Owned_Texture, texture_map: map[u32]u32)
{
    texture_map = make(map[u32]u32)

    
    for texture_idx in 0..<u32(len(data.textures)) {
        texture := &data.textures[texture_idx]
        if texture.source == nil {
            continue
        }
        
        image_idx := texture.source.?
        if int(image_idx) >= len(data.images) {
            continue
        }
        
        if existing_idx, found := texture_map[image_idx]; found {
            texture_map[texture_idx] = existing_idx
            continue
        }
        
        tex, ok := load_texture_from_gltf_image(data, image_idx, gltf_dir, upload_arena, cmd_buf)
        if !ok {
            continue
        }
        
        new_idx := u32(len(textures))
        append(&textures, tex)
        texture_map[texture_idx] = new_idx
        texture_map[image_idx] = new_idx
    }

    return textures, texture_map
}

load_scene_gltf :: proc(file_name: string, upload_arena: ^gpu.Arena, cmd_buf: gpu.Command_Buffer, load_textures: bool) -> Scene
{
    data, err_l := gltf2.load_from_file(file_name)
    switch err in err_l
    {
        case gltf2.JSON_Error: log.error(err)
        case gltf2.GLTF_Error: log.error(err)
    }
    defer gltf2.unload(data)

    gltf_dir := filepath.dir(file_name)

    textures: [dynamic]gpu.Owned_Texture
    texture_map: map[u32]u32
    if load_textures {
        textures, texture_map = load_textures_from_gltf(data, gltf_dir, upload_arena, cmd_buf)
        defer delete(texture_map)
    }

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
            uvs_final := uvs.([][2]f32)
            loaded := upload_mesh(upload_arena, cmd_buf, pos_final, normals_final, uvs_final, indices_u32[:])
            
            // Get texture IDs from material
            loaded.albedo_texture_id = nil
            loaded.normal_texture_id = nil
            
            if primitive.material != nil {
                material_idx := primitive.material.?
                if int(material_idx) < len(data.materials) {
                    material := &data.materials[material_idx]
                    
                    // Get albedo texture (baseColorTexture)
                    if material.metallic_roughness != nil {
                        if base_color_tex := material.metallic_roughness.?.base_color_texture; base_color_tex != nil {
                            gltf_tex_idx := base_color_tex.?.index
                            if our_tex_idx, found := texture_map[gltf_tex_idx]; found {
                                loaded.albedo_texture_id = our_tex_idx
                            }
                        }
                    }
                    
                    // Get normal texture
                    if normal_tex := material.normal_texture; normal_tex != nil {
                        gltf_tex_idx := normal_tex.?.index
                        if our_tex_idx, found := texture_map[gltf_tex_idx]; found {
                            loaded.normal_texture_id = our_tex_idx
                        }
                    }
                }
            }
            
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
        textures = textures,
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
