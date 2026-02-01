package shared

import "base:runtime"
import "core:fmt"

import log "core:log"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:slice"
import "gltf2"
import intr "base:intrinsics"

import sdl "vendor:sdl3"

MISSING_TEXTURE_ID :: 0

Texture_Type :: enum {
	Base_Color,
	Metallic_Roughness,
	Normal,
}

Gltf_Texture_Info :: struct {
	mesh_id:      u32,
	texture_type: Texture_Type,
	image_index:  int,
}

Mesh :: struct {
	pos:                    [dynamic][4]f32,
	normals:                [dynamic][4]f32,
	uvs:                    [dynamic][2]f32,
	indices:                [dynamic]u32,
	base_color_map:         u32,
	metallic_roughness_map: u32,
	normal_map:             u32,
}

destroy_mesh :: proc(mesh: ^Mesh) {
	delete(mesh.pos)
	delete(mesh.normals)
	delete(mesh.uvs)
	delete(mesh.indices)
	mesh^ = {}
}

Scene :: struct {
	meshes:    [dynamic]Mesh,
	instances: [dynamic]Instance,
}

destroy_scene :: proc(scene: ^Scene) {
	for &mesh in scene.meshes {
		destroy_mesh(&mesh)
	}

	delete(scene.meshes)
	delete(scene.instances)
	scene^ = {}
}

Instance :: struct {
	transform: matrix[4, 4]f32,
	mesh_idx:  u32,
}

// Input

Key_State :: struct {
	pressed:  bool,
	pressing: bool,
	released: bool,
}

Input :: struct {
	pressing_right_click: bool,
	left_click_pressed:   bool, // One-shot flag for left mouse button press
	keys:                 #sparse[sdl.Scancode]Key_State,
	mouse_dx:             f32, // pixels/dpi (inches), right is positive
	mouse_dy:             f32, // pixels/dpi (inches), up is positive
	mouse_wheel:          f32, // pixels/dpi (inches), up is positive
}

INPUT: Input

handle_window_events :: proc(window: ^sdl.Window) -> (proceed: bool) {
	// Reset "one-shot" inputs
	for &key in INPUT.keys {
		key.pressed = false
		key.released = false
	}
	INPUT.mouse_dx = 0
	INPUT.mouse_dy = 0
	INPUT.left_click_pressed = false
	INPUT.mouse_wheel = 0
	event: sdl.Event
	proceed = true
	for sdl.PollEvent(&event) {
		#partial switch event.type {
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
						INPUT.left_click_pressed = true
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

				if event.type == .KEY_DOWN {
					INPUT.keys[event.scancode].pressed = true
					INPUT.keys[event.scancode].pressing = true
				} else {
					INPUT.keys[event.scancode].pressing = false
					INPUT.keys[event.scancode].released = true
				}
			}
		case .MOUSE_MOTION:
			{
				event := event.motion
				INPUT.mouse_dx += event.xrel
				INPUT.mouse_dy -= event.yrel // In sdl, up is negative
			}
		case .MOUSE_WHEEL:
			{
				event := event.wheel
				INPUT.mouse_wheel += event.y
			}
		}
	}

	return
}

first_person_camera_view :: proc(delta_time: f32) -> matrix[4, 4]f32 {
	@(static) cam_pos: [3]f32 = {-7.581631, 1.1906259, 0.25928685}

	@(static) angle: [2]f32 = {1.570796, 0.3665192}

	cam_rot: quaternion128 = 1

	mouse_sensitivity := math.to_radians_f32(0.2) // Radians per pixel
	mouse: [2]f32
	if INPUT.pressing_right_click {
		mouse.x = INPUT.mouse_dx * mouse_sensitivity
		mouse.y = INPUT.mouse_dy * mouse_sensitivity
	}

	angle += mouse

	// Wrap angle.x
	for angle.x < 0 do angle.x += 2 * math.PI
	for angle.x > 2 * math.PI do angle.x -= 2 * math.PI

	angle.y = clamp(angle.y, math.to_radians_f32(-90), math.to_radians_f32(90))
	y_rot := linalg.quaternion_angle_axis(angle.y, [3]f32{-1, 0, 0})
	x_rot := linalg.quaternion_angle_axis(angle.x, [3]f32{0, 1, 0})
	cam_rot = x_rot * y_rot

	// Movement
	@(static) cur_vel: [3]f32
	move_speed: f32 : 6.0
	move_speed_fast: f32 : 15.0
	move_accel: f32 : 300.0

	keyboard_dir_xz: [3]f32
	keyboard_dir_y: f32
	if INPUT.pressing_right_click {
		keyboard_dir_xz.x = f32(int(INPUT.keys[.D].pressing) - int(INPUT.keys[.A].pressing))
		keyboard_dir_xz.z = f32(int(INPUT.keys[.W].pressing) - int(INPUT.keys[.S].pressing))
		keyboard_dir_y = f32(int(INPUT.keys[.E].pressing) - int(INPUT.keys[.Q].pressing))

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

	approach_linear :: proc(cur: [3]f32, target: [3]f32, delta: f32) -> [3]f32 {
		diff := target - cur
		dist := linalg.length(diff)

		if dist <= delta do return target
		return cur + diff / dist * delta
	}
}

// Orbit camera that rotates around a fixed origin point
orbit_camera_view :: proc(
	delta_time: f32,
	default_position: [3]f32 = {0, 0, 5},
	origin: [3]f32 = {0, 0, 0},
) -> matrix[4, 4]f32 {
	@(static) initialized: bool
	@(static) azimuth: f32 // Horizontal rotation angle (around Y axis)
	@(static) elevation: f32 // Vertical rotation angle (pitch)
	@(static) distance: f32 // Distance from origin

	if !initialized {
		// Calculate initial spherical coordinates from default position
		offset := default_position - origin
		distance = linalg.length(offset)

		if distance < 0.001 {
			// Default position is at origin, use default values
			distance = 5.0
			azimuth = 0.0
			elevation = 0.0
		} else {
			// Calculate azimuth (horizontal angle)
			azimuth = math.atan2(offset.x, offset.z)

			// Calculate elevation (vertical angle)
			horizontal_dist := math.sqrt(offset.x * offset.x + offset.z * offset.z)
			elevation = math.atan2(offset.y, horizontal_dist)
		}

		initialized = true
	}

	// Mouse input for rotation
	mouse_sensitivity := math.to_radians_f32(0.2) // Radians per pixel
	if INPUT.pressing_right_click {
		azimuth += INPUT.mouse_dx * mouse_sensitivity
		elevation -= INPUT.mouse_dy * mouse_sensitivity
	}

	// Wrap azimuth
	for azimuth < 0 do azimuth += 2 * math.PI
	for azimuth > 2 * math.PI do azimuth -= 2 * math.PI

	// Clamp elevation to prevent gimbal lock
	elevation = clamp(elevation, math.to_radians_f32(-89), math.to_radians_f32(89))

	// Mouse wheel input for zoom (distance adjustment) with interpolation
	@(static) target_distance: f32
	@(static) target_distance_initialized: bool
	if !target_distance_initialized {
		target_distance = distance
		target_distance_initialized = true
	}

	zoom_speed: f32 : 1.5 // Distance change per wheel tick
	zoom_factor: f32 : 4.0 // Linear interpolation factor (percentage per second)
	if INPUT.mouse_wheel != 0 {
		target_distance -= INPUT.mouse_wheel * zoom_speed
		target_distance = max(target_distance, 1.5) // Minimum distance
		target_distance = min(target_distance, 50.0) // Maximum distance (reasonable limit)
	}

	// Linear interpolation: speed proportional to current distance
	// Closer = slower zoom, further = faster zoom
	distance_diff := target_distance - distance
	if abs(distance_diff) > 0.001 {
		// Interpolate based on percentage of current distance
		max_change := distance * zoom_factor * delta_time
		if abs(distance_diff) <= max_change {
			distance = target_distance
		} else {
			distance += math.sign(distance_diff) * max_change
		}
	}

	// Calculate camera position from spherical coordinates
	cos_elev := math.cos(elevation)
	cam_pos :=
		origin +
		[3]f32 {
				distance * math.sin(azimuth) * cos_elev,
				distance * math.sin(elevation),
				distance * math.cos(azimuth) * cos_elev,
			}

	// Calculate forward direction from camera to origin to build quaternion
	forward := linalg.normalize(origin - cam_pos)

	// Calculate angles from forward vector (matching first-person camera convention)
	// azimuth (yaw) is rotation around Y axis
	forward_azimuth := math.atan2(forward.x, forward.z)

	// elevation (pitch) is rotation around X axis
	forward_horizontal_dist := math.sqrt(forward.x * forward.x + forward.z * forward.z)
	forward_elevation := math.atan2(forward.y, forward_horizontal_dist)

	// Build quaternion using same method as first-person camera
	cam_rot: quaternion128 = 1
	y_rot := linalg.quaternion_angle_axis(forward_elevation, [3]f32{-1, 0, 0})
	x_rot := linalg.quaternion_angle_axis(forward_azimuth, [3]f32{0, 1, 0})
	cam_rot = x_rot * y_rot

	return world_to_view_mat(cam_pos, cam_rot)
}

world_to_view_mat :: proc(cam_pos: [3]f32, cam_rot: quaternion128) -> matrix[4, 4]f32 {
	view_rot := linalg.normalize(linalg.quaternion_inverse(cam_rot))
	view_pos := -cam_pos
	return(
		#force_inline linalg.matrix4_from_quaternion(view_rot) *
		#force_inline linalg.matrix4_translate(view_pos) \
	)
}

// I/O

buffer_slice_with_stride :: proc(
	$T: typeid,
	data: ^gltf2.Data,
	accessor_index: gltf2.Integer,
	allocator := context.allocator,
) -> []T {
	accessor := data.accessors[accessor_index]
	assert(accessor.buffer_view != nil, "accessor must have buffer_view")

	buffer_view := data.buffer_views[accessor.buffer_view.?]

	if _, ok := accessor.sparse.?; ok {
		assert(false, "Sparse not supported")
		return nil
	}

	start_byte := accessor.byte_offset + buffer_view.byte_offset
	uri := data.buffers[buffer_view.buffer].uri

	buffer_data: []byte
	#partial switch v in uri {
	case []byte:
		buffer_data = v
	case string:
		assert(false, "URI string not supported")
		return nil
	}

	element_size := size_of(T)

	stride := int(buffer_view.byte_stride.? or_else gltf2.Integer(element_size))

	count := int(accessor.count)
	result_data := make([]T, count, allocator)

	for i in 0 ..< count {
		src_offset := int(start_byte) + i * stride
		assert(src_offset + element_size <= len(buffer_data), "buffer access out of bounds")
		mem.copy(&result_data[i], &buffer_data[src_offset], element_size)
	}

	return result_data
}

load_scene_gltf :: proc(
	contents: []byte,
) -> (
	Scene,
	[]Gltf_Texture_Info,
	^gltf2.Data,
) {
	options := gltf2.Options{}
	options.is_glb = true
	data, err_l := gltf2.parse(contents, options)
	switch err in err_l
	{
	case gltf2.JSON_Error:
		log.error(err)
	case gltf2.GLTF_Error:
		log.error(err)
	}
	// Note: data is returned to caller, who should call gltf2.unload(data) when done

	texture_infos: [dynamic]Gltf_Texture_Info

	log.info(fmt.tprintf("Collecting texture info from %v textures in GLTF", len(data.textures)))

	// Build a map from texture index to image index for quick lookup
	texture_to_image: map[int]int
	defer delete(texture_to_image)
	for texture, i in data.textures {
		if texture.source != nil {
			image_idx := texture.source.?
			if int(image_idx) >= len(data.images) {
				log.error(
					fmt.tprintf(
						"Texture %v references invalid image index %v (only %v images available)",
						i,
						image_idx,
						len(data.images),
					),
				)
				continue
			}
			texture_to_image[i] = int(image_idx)
		} else {
			log.info(fmt.tprintf("Texture %v has no source, skipping", i))
		}
	}

	// Load meshes
	meshes: [dynamic]Mesh
	start_idx: [dynamic]u32
	defer delete(start_idx)
	for mesh, i in data.meshes {
		append(&start_idx, u32(len(meshes)))

		for primitive, j in mesh.primitives {
			assert(primitive.mode == .Triangles)

			positions := buffer_slice_with_stride(
				[3]f32,
				data,
				primitive.attributes["POSITION"],
				context.temp_allocator,
			)
			normals := buffer_slice_with_stride(
				[3]f32,
				data,
				primitive.attributes["NORMAL"],
				context.temp_allocator,
			)
			uvs := buffer_slice_with_stride(
				[2]f32,
				data,
				primitive.attributes["TEXCOORD_0"],
				context.temp_allocator,
			)

			lm_uvs: [][2]f32
			if texcoord1_idx, ok := primitive.attributes["TEXCOORD_1"]; ok {
				lm_uvs = buffer_slice_with_stride(
					[2]f32,
					data,
					texcoord1_idx,
					context.temp_allocator,
				)
			}

			indices := gltf2.buffer_slice(data, primitive.indices.?)

			indices_u32: [dynamic]u32
			defer delete(indices_u32)
			#partial switch ids in indices
			{
			case []u16:
				for i in 0 ..< len(ids) do append(&indices_u32, u32(ids[i]))
			case []u32:
				for i in 0 ..< len(ids) do append(&indices_u32, ids[i])
			case:
				assert(false)
			}

			mesh_idx := u32(len(meshes))
			base_color_map: u32 = MISSING_TEXTURE_ID
			metallic_roughness_map: u32 = MISSING_TEXTURE_ID
			normal_map: u32 = MISSING_TEXTURE_ID

			if primitive.material != nil {
				material_idx := primitive.material.?
				material := data.materials[material_idx]

				// Base color texture
				if material.metallic_roughness != nil {
					if base_color_tex := material.metallic_roughness.?.base_color_texture;
					   base_color_tex != nil {
						if image_idx, ok := texture_to_image[int(base_color_tex.?.index)]; ok {
							append(
								&texture_infos,
								Gltf_Texture_Info {
									mesh_id = mesh_idx,
									texture_type = .Base_Color,
									image_index = image_idx,
								},
							)
						}
					}
					// Metallic roughness texture
					if mr_tex := material.metallic_roughness.?.metallic_roughness_texture;
					   mr_tex != nil {
						if image_idx, ok := texture_to_image[int(mr_tex.?.index)]; ok {
							append(
								&texture_infos,
								Gltf_Texture_Info {
									mesh_id = mesh_idx,
									texture_type = .Metallic_Roughness,
									image_index = image_idx,
								},
							)
						}
					}
				}

				// Normal texture
				if normal_tex := material.normal_texture; normal_tex != nil {
					if image_idx, ok := texture_to_image[int(normal_tex.?.index)]; ok {
						append(
							&texture_infos,
							Gltf_Texture_Info {
								mesh_id = mesh_idx,
								texture_type = .Normal,
								image_index = image_idx,
							},
						)
					}
				}
			}

			// Convert vec3 to vec4 (adding w=0 component)
			pos_final := to_vec4_array(positions, allocator = context.temp_allocator)
			normals_final := to_vec4_array(normals, allocator = context.temp_allocator)
			// Use TEXCOORD_0 if available, otherwise create default UVs
			uvs_final: [][2]f32
			if len(uvs) > 0 {
				uvs_final = uvs
			} else {
				// Create default UVs if not present
				uvs_final = make([][2]f32, len(positions), allocator = context.temp_allocator)
				for &uv, i in uvs_final {
					uv = {0.0, 0.0}
				}
			}

			loaded := Mesh {
				pos = slice.clone_to_dynamic(pos_final),
				normals = slice.clone_to_dynamic(normals_final),
				uvs = slice.clone_to_dynamic(uvs_final),
				indices = slice.clone_to_dynamic(indices_u32[:]),
				base_color_map = base_color_map,
				metallic_roughness_map = metallic_roughness_map,
				normal_map = normal_map,
			}
			append(&meshes, loaded)
		}
	}

	// Load instances
	instances: [dynamic]Instance
	for node_idx in data.scenes[0].nodes {
		node := data.nodes[node_idx]

		traverse_node(&instances, data, 1, int(node_idx), meshes, start_idx)

		traverse_node :: proc(
			instances: ^[dynamic]Instance,
			data: ^gltf2.Data,
			parent_transform: matrix[4, 4]f32,
			node_idx: int,
			meshes: [dynamic]Mesh,
			start_idx: [dynamic]u32,
		) {
			node := data.nodes[node_idx]

			flip_z: matrix[4, 4]f32 = 1
			flip_z[2, 2] = -1
			local_transform := xform_to_mat(node.translation, node.rotation, node.scale)
			transform := parent_transform * local_transform
			if node.mesh != nil {
				mesh_idx := node.mesh.?
				mesh := data.meshes[mesh_idx]

				for primitive, j in mesh.primitives {
					primitive_idx := start_idx[mesh_idx] + u32(j)
					instance := Instance {
						transform = flip_z * transform,
						mesh_idx  = primitive_idx,
					}
					append(instances, instance)
				}
			}

			for child in node.children {
				traverse_node(instances, data, transform, int(child), meshes, start_idx)
			}
		}
	}

	return {instances = instances, meshes = meshes}, texture_infos[:], data
}

to_vec4_array :: proc(array: [][3]f32, allocator: runtime.Allocator) -> [][4]f32 {
	res := make([][4]f32, len(array), allocator = allocator)
	for &v, i in res do v = {array[i].x, array[i].y, array[i].z, 0.0}
	return res
}

xform_to_mat_f64 :: proc(pos: [3]f64, rot: quaternion256, scale: [3]f64) -> matrix[4, 4]f32 {
	return(
		cast(matrix[4, 4]f32)(#force_inline linalg.matrix4_translate(pos) *
			#force_inline linalg.matrix4_from_quaternion(rot) *
			#force_inline linalg.matrix4_scale(scale)) \
	)
}

xform_to_mat_f32 :: proc(pos: [3]f32, rot: quaternion128, scale: [3]f32) -> matrix[4, 4]f32 {
	return(
		#force_inline linalg.matrix4_translate(pos) *
		#force_inline linalg.matrix4_from_quaternion(rot) *
		#force_inline linalg.matrix4_scale(scale) \
	)
}

xform_to_mat :: proc {
	xform_to_mat_f32,
	xform_to_mat_f64,
}

transform_to_gpu_transform :: proc(transform: matrix[4, 4]f32) -> [12]f32 {
	transform_row_major := intr.transpose(transform)
	flattened := linalg.matrix_flatten(transform_row_major)
	return [12]f32 { flattened[0], flattened[1], flattened[2], flattened[3], flattened[4], flattened[5], flattened[6], flattened[7], flattened[8], flattened[9], flattened[10], flattened[11], }
}
