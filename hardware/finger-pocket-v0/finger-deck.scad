// Concept v0; mm. Regenerate using dimensions.json and build.py.
translate([-0.3, 27.7, 12.0]) {
rotate([180, 0, 0]) {
difference() {
difference() {
difference() {
difference() {
difference() {
translate([0.3, 2.3, 10.0]) {
cube([56.4, 25.4, 2.0]);
}
translate([11.7, 1.7, 9.95]) {
cube([3.6, 3.1, 2.1]);
}
}
translate([44.7, 25.2, 9.95]) {
cube([3.6, 3.1, 2.1]);
}
}
translate([28.0, 12.0, 9.95]) {
hull() {
union() {
union() {
union() {
translate([1, 1, 0]) {
cylinder(h=2.1,r=1,$fn=24);
}
translate([1, 5.0, 0]) {
cylinder(h=2.1,r=1,$fn=24);
}
}
translate([7.0, 1, 0]) {
cylinder(h=2.1,r=1,$fn=24);
}
}
translate([7.0, 5.0, 0]) {
cylinder(h=2.1,r=1,$fn=24);
}
}
}
}
}
translate([21.7, 6.7, 9.95]) {
cube([20.6, 16.6, 0.55]);
}
}
hull() {
union() {
translate([28.0, 12.0, 11.5]) {
hull() {
union() {
union() {
union() {
translate([1, 1, 0]) {
cylinder(h=0.05,r=1,$fn=24);
}
translate([1, 5.0, 0]) {
cylinder(h=0.05,r=1,$fn=24);
}
}
translate([7.0, 1, 0]) {
cylinder(h=0.05,r=1,$fn=24);
}
}
translate([7.0, 5.0, 0]) {
cylinder(h=0.05,r=1,$fn=24);
}
}
}
}
translate([27.5, 11.5, 12.05]) {
hull() {
union() {
union() {
union() {
translate([1.5, 1.5, 0]) {
cylinder(h=0.05,r=1.5,$fn=24);
}
translate([1.5, 5.5, 0]) {
cylinder(h=0.05,r=1.5,$fn=24);
}
}
translate([7.5, 1.5, 0]) {
cylinder(h=0.05,r=1.5,$fn=24);
}
}
translate([7.5, 5.5, 0]) {
cylinder(h=0.05,r=1.5,$fn=24);
}
}
}
}
}
}
}
}
}
