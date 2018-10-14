#version 330

uniform mat4 modelViewProjection;

in vec3 vPosition;
in vec3 vTexcoord;
in vec3 vNormal;

out vec3 fPosition;
out vec3 fTexcoord;
out vec3 fNormal;

void main() {
	fPosition = vPosition;
	fTexcoord = vTexcoord;
	fNormal = vNormal;

	gl_Position = modelViewProjection * vec4(fPosition, 1.0);
	gl_Position.z *= 0.01;
}
