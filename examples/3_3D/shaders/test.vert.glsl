#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) out vec4 _res_out_loc0_;

layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_slice_vec3;
layout(buffer_reference) readonly buffer _res_slice_vec4;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Data
{
    mat4 model_to_world;
    mat4 model_to_world_normal;
    mat4 world_to_view;
    mat4 view_to_proj;
    _res_slice_vec3 positions;
    _res_slice_vec4 normals;
};

struct Output
{
    vec4 pos;
    vec4 color;
};

void main();
layout(buffer_reference, std140) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, std140) readonly buffer _res_slice_vec3 { vec3 _res_; };
layout(buffer_reference, std140) readonly buffer _res_slice_vec4 { vec4 _res_; };
layout(buffer_reference, std140) readonly buffer _res_ptr_Data { Data _res_; };

layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform writeonly image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];

layout(push_constant, std140) uniform Push
{
    _res_ptr_Data _res_vert_data_;
    _res_ptr_Data _res_frag_data_;
    _res_ptr_void _res_vert_indirect_data_;
    _res_ptr_void _res_frag_indirect_data_;
};

void main()
{
    uint vert_id = gl_VertexIndex;
    _res_ptr_Data data = _res_vert_data_;
    vec4 clip_pos;
    vec4 world_normal;
    Output vert_out;
    clip_pos = vec4(data._res_.positions[vert_id]._res_.xyz, 1.0);
    clip_pos = data._res_.model_to_world*clip_pos;
    clip_pos = data._res_.world_to_view*clip_pos;
    clip_pos = data._res_.view_to_proj*clip_pos;
    clip_pos.y = 0.0-clip_pos.y;
    world_normal = data._res_.model_to_world_normal*data._res_.normals[vert_id]._res_;
    vert_out.pos = clip_pos;
    vert_out.color = world_normal;
    gl_Position = vert_out.pos; _res_out_loc0_ = vert_out.color; ;
}

