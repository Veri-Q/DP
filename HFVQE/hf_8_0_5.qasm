// Generated from Cirq v1.0.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0, 0), q(0, 1), q(0, 2), q(0, 3), q(0, 4), q(0, 5), q(0, 6), q(0, 7)]
qreg q[8];


x q[0];
x q[1];
x q[2];
x q[3];

// Gate: ISWAP**0.5
cx q[3],q[4];
h q[3];
cx q[4],q[3];
t q[3];
cx q[4],q[3];
tdg q[3];
h q[3];
cx q[3],q[4];

rz(pi*1.5) q[3];
rz(pi*-0.5) q[4];

// Gate: ISWAP**0.5
cx q[3],q[4];
h q[3];
cx q[4],q[3];
t q[3];
cx q[4],q[3];
tdg q[3];
h q[3];
cx q[3],q[4];

rz(pi*1.0) q[3];

// Gate: ISWAP**0.5
cx q[4],q[5];
h q[4];
cx q[5],q[4];
t q[4];
cx q[5],q[4];
tdg q[4];
h q[4];
cx q[4],q[5];

// Gate: ISWAP**0.5
cx q[2],q[3];
h q[2];
cx q[3],q[2];
t q[2];
cx q[3],q[2];
tdg q[2];
h q[2];
cx q[2],q[3];

rz(pi*0.961542381) q[4];
rz(pi*0.038457619) q[5];
rz(pi*1.2095027292) q[2];
rz(pi*-0.2095027292) q[3];

// Gate: ISWAP**0.5
cx q[4],q[5];
h q[4];
cx q[5],q[4];
t q[4];
cx q[5],q[4];
tdg q[4];
h q[4];
cx q[4],q[5];

// Gate: ISWAP**0.5
cx q[2],q[3];
h q[2];
cx q[3],q[2];
t q[2];
cx q[3],q[2];
tdg q[2];
h q[2];
cx q[2],q[3];

rz(pi*1.0) q[4];

// Gate: ISWAP**0.5
cx q[5],q[6];
h q[5];
cx q[6],q[5];
t q[5];
cx q[6],q[5];
tdg q[5];
h q[5];
cx q[5],q[6];

rz(pi*1.0) q[2];

// Gate: ISWAP**0.5
cx q[3],q[4];
h q[3];
cx q[4],q[3];
t q[3];
cx q[4],q[3];
tdg q[3];
h q[3];
cx q[3],q[4];

rz(pi*1.5) q[5];
rz(pi*-0.5) q[6];

// Gate: ISWAP**0.5
cx q[1],q[2];
h q[1];
cx q[2],q[1];
t q[1];
cx q[2],q[1];
tdg q[1];
h q[1];
cx q[1],q[2];

rz(pi*1.5) q[3];
rz(pi*-0.5) q[4];

// Gate: ISWAP**0.5
cx q[5],q[6];
h q[5];
cx q[6],q[5];
t q[5];
cx q[6],q[5];
tdg q[5];
h q[5];
cx q[5],q[6];

rz(pi*1.5) q[1];
rz(pi*-0.5) q[2];

// Gate: ISWAP**0.5
cx q[3],q[4];
h q[3];
cx q[4],q[3];
t q[3];
cx q[4],q[3];
tdg q[3];
h q[3];
cx q[3],q[4];

rz(pi*1.0) q[5];

// Gate: ISWAP**0.5
cx q[6],q[7];
h q[6];
cx q[7],q[6];
t q[6];
cx q[7],q[6];
tdg q[6];
h q[6];
cx q[6],q[7];

// Gate: ISWAP**0.5
cx q[1],q[2];
h q[1];
cx q[2],q[1];
t q[1];
cx q[2],q[1];
tdg q[1];
h q[1];
cx q[1],q[2];

rz(pi*1.0) q[3];

// Gate: ISWAP**0.5
cx q[4],q[5];
h q[4];
cx q[5],q[4];
t q[4];
cx q[5],q[4];
tdg q[4];
h q[4];
cx q[4],q[5];

rz(pi*1.0871215972) q[6];
rz(pi*-0.0871215972) q[7];
rz(pi*1.0) q[1];

// Gate: ISWAP**0.5
cx q[2],q[3];
h q[2];
cx q[3],q[2];
t q[2];
cx q[3],q[2];
tdg q[2];
h q[2];
cx q[2],q[3];

rz(pi*1.036430841) q[4];
rz(pi*-0.036430841) q[5];

// Gate: ISWAP**0.5
cx q[6],q[7];
h q[6];
cx q[7],q[6];
t q[6];
cx q[7],q[6];
tdg q[6];
h q[6];
cx q[6],q[7];

// Gate: ISWAP**0.5
cx q[0],q[1];
h q[0];
cx q[1],q[0];
t q[0];
cx q[1],q[0];
tdg q[0];
h q[0];
cx q[0],q[1];

rz(pi*0.9664466832) q[2];
rz(pi*0.0335533168) q[3];

// Gate: ISWAP**0.5
cx q[4],q[5];
h q[4];
cx q[5],q[4];
t q[4];
cx q[5],q[4];
tdg q[4];
h q[4];
cx q[4],q[5];

rz(pi*1.0) q[6];
rz(pi*1.0251681994) q[0];
rz(pi*-0.0251681994) q[1];

// Gate: ISWAP**0.5
cx q[2],q[3];
h q[2];
cx q[3],q[2];
t q[2];
cx q[3],q[2];
tdg q[2];
h q[2];
cx q[2],q[3];

rz(pi*1.0) q[4];

// Gate: ISWAP**0.5
cx q[5],q[6];
h q[5];
cx q[6],q[5];
t q[5];
cx q[6],q[5];
tdg q[5];
h q[5];
cx q[5],q[6];

// Gate: ISWAP**0.5
cx q[0],q[1];
h q[0];
cx q[1],q[0];
t q[0];
cx q[1],q[0];
tdg q[0];
h q[0];
cx q[0],q[1];

rz(pi*1.0) q[2];

// Gate: ISWAP**0.5
cx q[3],q[4];
h q[3];
cx q[4],q[3];
t q[3];
cx q[4],q[3];
tdg q[3];
h q[3];
cx q[3],q[4];

rz(pi*1.5) q[5];
rz(pi*-0.5) q[6];
rz(pi*1.0) q[0];

// Gate: ISWAP**0.5
cx q[1],q[2];
h q[1];
cx q[2],q[1];
t q[1];
cx q[2],q[1];
tdg q[1];
h q[1];
cx q[1],q[2];

rz(pi*1.5) q[3];
rz(pi*-0.5) q[4];

// Gate: ISWAP**0.5
cx q[5],q[6];
h q[5];
cx q[6],q[5];
t q[5];
cx q[6],q[5];
tdg q[5];
h q[5];
cx q[5],q[6];

rz(pi*1.5) q[1];
rz(pi*-0.5) q[2];

// Gate: ISWAP**0.5
cx q[3],q[4];
h q[3];
cx q[4],q[3];
t q[3];
cx q[4],q[3];
tdg q[3];
h q[3];
cx q[3],q[4];

rz(pi*1.0) q[5];

// Gate: ISWAP**0.5
cx q[1],q[2];
h q[1];
cx q[2],q[1];
t q[1];
cx q[2],q[1];
tdg q[1];
h q[1];
cx q[1],q[2];

rz(pi*1.0) q[3];

// Gate: ISWAP**0.5
cx q[4],q[5];
h q[4];
cx q[5],q[4];
t q[4];
cx q[5],q[4];
tdg q[4];
h q[4];
cx q[4],q[5];

rz(pi*1.0) q[1];

// Gate: ISWAP**0.5
cx q[2],q[3];
h q[2];
cx q[3],q[2];
t q[2];
cx q[3],q[2];
tdg q[2];
h q[2];
cx q[2],q[3];

rz(pi*0.9347215966) q[4];
rz(pi*0.0652784034) q[5];
rz(pi*0.9294444214) q[2];
rz(pi*0.0705555786) q[3];

// Gate: ISWAP**0.5
cx q[4],q[5];
h q[4];
cx q[5],q[4];
t q[4];
cx q[5],q[4];
tdg q[4];
h q[4];
cx q[4],q[5];

// Gate: ISWAP**0.5
cx q[2],q[3];
h q[2];
cx q[3],q[2];
t q[2];
cx q[3],q[2];
tdg q[2];
h q[2];
cx q[2],q[3];

rz(pi*1.0) q[4];
rz(pi*1.0) q[2];

// Gate: ISWAP**0.5
cx q[3],q[4];
h q[3];
cx q[4],q[3];
t q[3];
cx q[4],q[3];
tdg q[3];
h q[3];
cx q[3],q[4];

rz(pi*1.5) q[3];
rz(pi*-0.5) q[4];

// Gate: ISWAP**0.5
cx q[3],q[4];
h q[3];
cx q[4],q[3];
t q[3];
cx q[4],q[3];
tdg q[3];
h q[3];
cx q[3],q[4];

rz(pi*1.0) q[3];