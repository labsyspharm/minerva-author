#version 300 es
precision highp int;
precision highp float;
precision highp usampler2D;

uniform usampler2D u_tile;
uniform vec3 u_tile_color;
uniform vec2 u_tile_range;

uniform uint u8;

in vec2 uv;
out vec4 color;


void main() {

  uvec2 pixel = texture(u_tile, uv).rg;
  float value = float(pixel.r * u8 + pixel.g) / 65535.;

  float min_ = u_tile_range[0];
  float max_ = u_tile_range[1];

  // Threshhold pixel within range
  float pixel_val = clamp((value - min_) / (max_ - min_), 0.0, 1.0);

  // Color pixel value
  vec3 pixel_color = pixel_val * u_tile_color;
  color = vec4(pixel_color, 1.0);
}
