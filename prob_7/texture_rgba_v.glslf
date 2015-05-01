////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rgba texture mapping of viewport span, fragment shader
////////////////////////////////////////////////////////////////////////////////////////////////////////////

out uvec4 xx_FragColor;

uniform usampler2DRect albedo_map;


void main()
{
	uvec4 tex00 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0, -7.0));
	uvec4 tex01 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0, -6.0));
	uvec4 tex02 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0, -5.0));
	uvec4 tex03 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0, -4.0));
	uvec4 tex04 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0, -3.0));
	uvec4 tex05 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0, -2.0));
	uvec4 tex06 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0, -1.0));
	uvec4 tex07 = texture(albedo_map, gl_FragCoord.xy);
	uvec4 tex08 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0,  1.0));
	uvec4 tex09 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0,  2.0));
	uvec4 tex10 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0,  3.0));
	uvec4 tex11 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0,  4.0));
	uvec4 tex12 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0,  5.0));
	uvec4 tex13 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0,  6.0));
	uvec4 tex14 = texture(albedo_map, gl_FragCoord.xy + vec2(0.0,  7.0));

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

	vec3 rgb = sum;

	xx_FragColor = uvec4(uvec3(rgb), tex07.w);
}
