////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rgba texture mapping of viewport span, fragment shader
////////////////////////////////////////////////////////////////////////////////////////////////////////////

out vec4 xx_FragColor;

uniform usampler2DRect albedo_map;
uniform usampler2DRect fbo_map;
uniform vec3 contrast_mid_k_cookie;


void main()
{
	uvec4 tex = texture(albedo_map, gl_FragCoord.xy);

	uvec4 tex00 = texture(fbo_map, gl_FragCoord.xy + vec2(-7.0, 0.0));
	uvec4 tex01 = texture(fbo_map, gl_FragCoord.xy + vec2(-6.0, 0.0));
	uvec4 tex02 = texture(fbo_map, gl_FragCoord.xy + vec2(-5.0, 0.0));
	uvec4 tex03 = texture(fbo_map, gl_FragCoord.xy + vec2(-4.0, 0.0));
	uvec4 tex04 = texture(fbo_map, gl_FragCoord.xy + vec2(-3.0, 0.0));
	uvec4 tex05 = texture(fbo_map, gl_FragCoord.xy + vec2(-2.0, 0.0));
	uvec4 tex06 = texture(fbo_map, gl_FragCoord.xy + vec2(-1.0, 0.0));
	uvec4 tex07 = texture(fbo_map, gl_FragCoord.xy);
	uvec4 tex08 = texture(fbo_map, gl_FragCoord.xy + vec2( 1.0, 0.0));
	uvec4 tex09 = texture(fbo_map, gl_FragCoord.xy + vec2( 2.0, 0.0));
	uvec4 tex10 = texture(fbo_map, gl_FragCoord.xy + vec2( 3.0, 0.0));
	uvec4 tex11 = texture(fbo_map, gl_FragCoord.xy + vec2( 4.0, 0.0));
	uvec4 tex12 = texture(fbo_map, gl_FragCoord.xy + vec2( 5.0, 0.0));
	uvec4 tex13 = texture(fbo_map, gl_FragCoord.xy + vec2( 6.0, 0.0));
	uvec4 tex14 = texture(fbo_map, gl_FragCoord.xy + vec2( 7.0, 0.0));

	vec3 sum = tex07.xyz * vec3(0.159576912161);
	sum += tex00.xyz * vec3(0.0044299121055113265);
	sum += tex01.xyz * vec3(0.00895781211794);
	sum += tex02.xyz * vec3(0.0215963866053);
	sum += tex03.xyz * vec3(0.0443683338718);
	sum += tex04.xyz * vec3(0.0776744219933);
	sum += tex05.xyz * vec3(0.115876621105);
	sum += tex06.xyz * vec3(0.147308056121);
	sum += tex08.xyz * vec3(0.147308056121);
	sum += tex09.xyz * vec3(0.115876621105);
	sum += tex10.xyz * vec3(0.0776744219933);
	sum += tex11.xyz * vec3(0.0443683338718);
	sum += tex12.xyz * vec3(0.0215963866053);
	sum += tex13.xyz * vec3(0.00895781211794);
	sum += tex14.xyz * vec3(0.0044299121055113265);

	vec3 rgb = sum * (1.0 / 255.0);

	// contrast adjustment
	rgb = (rgb - contrast_mid_k_cookie.x) * contrast_mid_k_cookie.y + contrast_mid_k_cookie.x;

	vec3 rgb_alt = tex.xyz * vec3(1.0 / 255.0);

	xx_FragColor = vec4(mix(rgb * rgb, rgb_alt, step(gl_FragCoord.x, 256.0 + 256.0 * contrast_mid_k_cookie.z)), 1.0);
}
