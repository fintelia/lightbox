#version 330

uniform mat4 modelViewProjection;
uniform sampler2DArray uTexture;


in vec3 fPosition;
in vec3 fTexcoord;
in vec3 fNormal;

out vec4 OutColor;

void main() {
	OutColor = texture(uTexture, fTexcoord);
	if(OutColor.a < 0.5)
		discard;
	OutColor.a = 1.0;
}
