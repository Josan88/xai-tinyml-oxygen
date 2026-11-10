// Concept v0; mm. Regenerate using dimensions.json and build.py.
translate([0, 30.0, 40.4]) {
rotate([180, 0, 0]) {
intersection() {
difference() {
difference() {
difference() {
difference() {
union() {
union() {
union() {
translate([0, 0, 38]) {
hull() {
union() {
union() {
union() {
translate([2, 2, 0]) {
cylinder(h=2.4,r=2,$fn=24);
}
translate([2, 88.0, 0]) {
cylinder(h=2.4,r=2,$fn=24);
}
}
translate([105.0, 2, 0]) {
cylinder(h=2.4,r=2,$fn=24);
}
}
translate([105.0, 88.0, 0]) {
cylinder(h=2.4,r=2,$fn=24);
}
}
}
}
translate([0, 2.3, 34.0]) {
cube([56.7, 25.4, 4.05]);
}
}
translate([53.7, 2.3, 12.0]) {
cube([3, 1.5, 26.05]);
}
}
translate([53.7, 26.2, 12.0]) {
cube([3, 1.5, 26.05]);
}
}
translate([6, 82.0, 37.95]) {
cylinder(h=2.5,r=1.7,$fn=48);
}
}
translate([101.0, 82.0, 37.95]) {
cylinder(h=2.5,r=1.7,$fn=48);
}
}
translate([65.0, 7, 37.95]) {
cylinder(h=2.5,r=1.7,$fn=48);
}
}
translate([101.0, 7, 37.95]) {
cylinder(h=2.5,r=1.7,$fn=48);
}
}
translate([0, 0, 0]) {
cube([59.0, 30.0, 41.4]);
}
}
}
}
