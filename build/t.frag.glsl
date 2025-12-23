#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

layout(location = 0) out vec4 _res_out_loc0_;
layout(location = 0) in vec4 _res_in_loc0_;

layout(buffer_reference) readonly buffer _res_ptr_void;

layout(buffer_reference, std140) readonly buffer _res_ptr_void { uint _res_void_; };

layout(push_constant, std140) uniform Push
{
    _res_ptr_void _res_vert_data_;
    _res_ptr_void _res_data_;
};

void main()
{
    vec4 color = _res_in_loc0_;
    _res_out_loc0_ = color;
}

