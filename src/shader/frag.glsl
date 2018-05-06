#version 330

uniform mat4 modelViewProjection;

in vec3 fPosition;
in vec2 fTexcoord;
in vec3 fNormal;

out vec4 OutColor;

void main() {
	OutColor = vec4(1, 0, 0, 1);
}
