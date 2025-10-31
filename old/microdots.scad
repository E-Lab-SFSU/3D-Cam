mm_per_dot=.5;
size=25.4;
for (n=[0:size/mm_per_dot]){
    for (m=[0:size/mm_per_dot])
    translate([n*mm_per_dot, m*mm_per_dot])
    circle(d=.001*25.4);
}

echo("DPI=", 1/(mm_per_dot/25.4));