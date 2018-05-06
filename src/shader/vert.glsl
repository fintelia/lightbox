#version 330

uniform mat4 modelViewProjection;

in vec3 vPosition;
in vec2 vTexcoord;
in vec3 vNormal;

out vec3 fPosition;
out vec2 fTexcoord;
out vec3 fNormal;

void main() {
	fPosition = vPosition;
	fTexcoord = vTexcoord;
	fNormal = vNormal;

	gl_Position = modelViewProjection * vec4(fPosition, 1.0);
}
