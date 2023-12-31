// Generated from Cirq v1.2.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0, 0), q(0, 1), q(0, 2), q(0, 3), q(0, 4), q(0, 5), q(0, 6), q(1, 0), q(1, 1), q(1, 2), q(1, 3), q(1, 4), q(1, 5), q(1, 6), q(2, 0), q(2, 1), q(2, 2), q(2, 3), q(2, 4), q(2, 5), q(2, 6)]
qreg q[21];


h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[0];
rz(pi*0.495) q[1];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[0];
u3(pi*0.5,0,pi*1.6637118504) q[1];
sx q[0];
cx q[0],q[1];
rx(pi*0.005) q[0];
ry(pi*0.5) q[1];
cx q[1],q[0];
sxdg q[1];
s q[1];
cx q[0],q[1];
u3(pi*0.5,pi*1.8412881496,0) q[0];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[1];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[2];
rz(pi*0.495) q[3];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[2];
u3(pi*0.5,0,pi*1.6637118504) q[3];
sx q[2];
cx q[2],q[3];
rx(pi*0.005) q[2];
ry(pi*0.5) q[3];
cx q[3],q[2];
sxdg q[3];
s q[3];
cx q[2],q[3];
u3(pi*0.5,pi*1.8412881496,0) q[2];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[3];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[4];
rz(pi*0.495) q[5];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[4];
u3(pi*0.5,0,pi*1.6637118504) q[5];
sx q[4];
cx q[4],q[5];
rx(pi*0.005) q[4];
ry(pi*0.5) q[5];
cx q[5],q[4];
sxdg q[5];
s q[5];
cx q[4],q[5];
u3(pi*0.5,pi*1.8412881496,0) q[4];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[5];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[7];
rz(pi*-0.495) q[8];
u3(pi*0.5,0,pi*0.8362881496) q[7];
u3(pi*0.5,0,pi*0.8362881496) q[8];
sx q[7];
cx q[7],q[8];
rx(pi*0.005) q[7];
ry(pi*0.5) q[8];
cx q[8],q[7];
sxdg q[8];
s q[8];
cx q[7],q[8];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[7];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[8];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[9];
rz(pi*-0.495) q[10];
u3(pi*0.5,0,pi*0.8362881496) q[9];
u3(pi*0.5,0,pi*0.8362881496) q[10];
sx q[9];
cx q[9],q[10];
rx(pi*0.005) q[9];
ry(pi*0.5) q[10];
cx q[10],q[9];
sxdg q[10];
s q[10];
cx q[9],q[10];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[9];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[10];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[11];
rz(pi*0.495) q[12];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[11];
u3(pi*0.5,0,pi*1.6637118504) q[12];
sx q[11];
cx q[11],q[12];
rx(pi*0.005) q[11];
ry(pi*0.5) q[12];
cx q[12],q[11];
sxdg q[12];
s q[12];
cx q[11],q[12];
u3(pi*0.5,pi*1.8412881496,0) q[11];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[12];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[14];
rz(pi*0.495) q[15];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[14];
u3(pi*0.5,0,pi*1.6637118504) q[15];
sx q[14];
cx q[14],q[15];
rx(pi*0.005) q[14];
ry(pi*0.5) q[15];
cx q[15],q[14];
sxdg q[15];
s q[15];
cx q[14],q[15];
u3(pi*0.5,pi*1.8412881496,0) q[14];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[15];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[16];
rz(pi*0.495) q[17];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[16];
u3(pi*0.5,0,pi*1.6637118504) q[17];
sx q[16];
cx q[16],q[17];
rx(pi*0.005) q[16];
ry(pi*0.5) q[17];
cx q[17],q[16];
sxdg q[17];
s q[17];
cx q[16],q[17];
u3(pi*0.5,pi*1.8412881496,0) q[16];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[17];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[18];
rz(pi*0.495) q[19];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[18];
u3(pi*0.5,0,pi*1.6637118504) q[19];
sx q[18];
cx q[18],q[19];
rx(pi*0.005) q[18];
ry(pi*0.5) q[19];
cx q[19],q[18];
sxdg q[19];
s q[19];
cx q[18],q[19];
u3(pi*0.5,pi*1.8412881496,0) q[18];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[19];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[1];
rz(pi*-0.495) q[2];
u3(pi*0.5,0,pi*0.8362881496) q[1];
u3(pi*0.5,0,pi*0.8362881496) q[2];
sx q[1];
cx q[1],q[2];
rx(pi*0.005) q[1];
ry(pi*0.5) q[2];
cx q[2],q[1];
sxdg q[2];
s q[2];
cx q[1],q[2];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[1];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[2];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[3];
rz(pi*0.495) q[4];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[3];
u3(pi*0.5,0,pi*1.6637118504) q[4];
sx q[3];
cx q[3],q[4];
rx(pi*0.005) q[3];
ry(pi*0.5) q[4];
cx q[4],q[3];
sxdg q[4];
s q[4];
cx q[3],q[4];
u3(pi*0.5,pi*1.8412881496,0) q[3];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[4];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[5];
rz(pi*-0.495) q[6];
u3(pi*0.5,0,pi*0.8362881496) q[5];
u3(pi*0.5,0,pi*0.8362881496) q[6];
sx q[5];
cx q[5],q[6];
rx(pi*0.005) q[5];
ry(pi*0.5) q[6];
cx q[6],q[5];
sxdg q[6];
s q[6];
cx q[5],q[6];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[5];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[6];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[8];
rz(pi*-0.495) q[9];
u3(pi*0.5,0,pi*0.8362881496) q[8];
u3(pi*0.5,0,pi*0.8362881496) q[9];
sx q[8];
cx q[8],q[9];
rx(pi*0.005) q[8];
ry(pi*0.5) q[9];
cx q[9],q[8];
sxdg q[9];
s q[9];
cx q[8],q[9];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[8];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[9];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[10];
rz(pi*-0.495) q[11];
u3(pi*0.5,0,pi*0.8362881496) q[10];
u3(pi*0.5,0,pi*0.8362881496) q[11];
sx q[10];
cx q[10],q[11];
rx(pi*0.005) q[10];
ry(pi*0.5) q[11];
cx q[11],q[10];
sxdg q[11];
s q[11];
cx q[10],q[11];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[10];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[11];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[12];
rz(pi*-0.495) q[13];
u3(pi*0.5,0,pi*0.8362881496) q[12];
u3(pi*0.5,0,pi*0.8362881496) q[13];
sx q[12];
cx q[12],q[13];
rx(pi*0.005) q[12];
ry(pi*0.5) q[13];
cx q[13],q[12];
sxdg q[13];
s q[13];
cx q[12],q[13];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[12];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[13];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[15];
rz(pi*0.495) q[16];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[15];
u3(pi*0.5,0,pi*1.6637118504) q[16];
sx q[15];
cx q[15],q[16];
rx(pi*0.005) q[15];
ry(pi*0.5) q[16];
cx q[16],q[15];
sxdg q[16];
s q[16];
cx q[15],q[16];
u3(pi*0.5,pi*1.8412881496,0) q[15];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[16];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[17];
rz(pi*-0.495) q[18];
u3(pi*0.5,0,pi*0.8362881496) q[17];
u3(pi*0.5,0,pi*0.8362881496) q[18];
sx q[17];
cx q[17],q[18];
rx(pi*0.005) q[17];
ry(pi*0.5) q[18];
cx q[18],q[17];
sxdg q[18];
s q[18];
cx q[17],q[18];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[17];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[18];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[19];
rz(pi*-0.495) q[20];
u3(pi*0.5,0,pi*0.8362881496) q[19];
u3(pi*0.5,0,pi*0.8362881496) q[20];
sx q[19];
cx q[19],q[20];
rx(pi*0.005) q[19];
ry(pi*0.5) q[20];
cx q[20],q[19];
sxdg q[20];
s q[20];
cx q[19],q[20];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[19];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[20];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[0];
rz(pi*-0.495) q[7];
u3(pi*0.5,0,pi*0.8362881496) q[0];
u3(pi*0.5,0,pi*0.8362881496) q[7];
sx q[0];
cx q[0],q[7];
rx(pi*0.005) q[0];
ry(pi*0.5) q[7];
cx q[7],q[0];
sxdg q[7];
s q[7];
cx q[0],q[7];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[0];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[7];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[1];
rz(pi*0.495) q[8];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[1];
u3(pi*0.5,0,pi*1.6637118504) q[8];
sx q[1];
cx q[1],q[8];
rx(pi*0.005) q[1];
ry(pi*0.5) q[8];
cx q[8],q[1];
sxdg q[8];
s q[8];
cx q[1],q[8];
u3(pi*0.5,pi*1.8412881496,0) q[1];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[8];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[2];
rz(pi*0.495) q[9];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[2];
u3(pi*0.5,0,pi*1.6637118504) q[9];
sx q[2];
cx q[2],q[9];
rx(pi*0.005) q[2];
ry(pi*0.5) q[9];
cx q[9],q[2];
sxdg q[9];
s q[9];
cx q[2],q[9];
u3(pi*0.5,pi*1.8412881496,0) q[2];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[9];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[3];
rz(pi*0.495) q[10];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[3];
u3(pi*0.5,0,pi*1.6637118504) q[10];
sx q[3];
cx q[3],q[10];
rx(pi*0.005) q[3];
ry(pi*0.5) q[10];
cx q[10],q[3];
sxdg q[10];
s q[10];
cx q[3],q[10];
u3(pi*0.5,pi*1.8412881496,0) q[3];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[10];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[4];
rz(pi*0.495) q[11];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[4];
u3(pi*0.5,0,pi*1.6637118504) q[11];
sx q[4];
cx q[4],q[11];
rx(pi*0.005) q[4];
ry(pi*0.5) q[11];
cx q[11],q[4];
sxdg q[11];
s q[11];
cx q[4],q[11];
u3(pi*0.5,pi*1.8412881496,0) q[4];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[11];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[5];
rz(pi*0.495) q[12];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[5];
u3(pi*0.5,0,pi*1.6637118504) q[12];
sx q[5];
cx q[5],q[12];
rx(pi*0.005) q[5];
ry(pi*0.5) q[12];
cx q[12],q[5];
sxdg q[12];
s q[12];
cx q[5],q[12];
u3(pi*0.5,pi*1.8412881496,0) q[5];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[12];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[6];
rz(pi*-0.495) q[13];
u3(pi*0.5,0,pi*0.8362881496) q[6];
u3(pi*0.5,0,pi*0.8362881496) q[13];
sx q[6];
cx q[6],q[13];
rx(pi*0.005) q[6];
ry(pi*0.5) q[13];
cx q[13],q[6];
sxdg q[13];
s q[13];
cx q[6],q[13];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[6];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[13];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[7];
rz(pi*0.495) q[14];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[7];
u3(pi*0.5,0,pi*1.6637118504) q[14];
sx q[7];
cx q[7],q[14];
rx(pi*0.005) q[7];
ry(pi*0.5) q[14];
cx q[14],q[7];
sxdg q[14];
s q[14];
cx q[7],q[14];
u3(pi*0.5,pi*1.8412881496,0) q[7];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[14];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[8];
rz(pi*-0.495) q[15];
u3(pi*0.5,0,pi*0.8362881496) q[8];
u3(pi*0.5,0,pi*0.8362881496) q[15];
sx q[8];
cx q[8],q[15];
rx(pi*0.005) q[8];
ry(pi*0.5) q[15];
cx q[15],q[8];
sxdg q[15];
s q[15];
cx q[8],q[15];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[8];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[15];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[9];
rz(pi*-0.495) q[16];
u3(pi*0.5,0,pi*0.8362881496) q[9];
u3(pi*0.5,0,pi*0.8362881496) q[16];
sx q[9];
cx q[9],q[16];
rx(pi*0.005) q[9];
ry(pi*0.5) q[16];
cx q[16],q[9];
sxdg q[16];
s q[16];
cx q[9],q[16];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[9];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[16];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[10];
rz(pi*0.495) q[17];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[10];
u3(pi*0.5,0,pi*1.6637118504) q[17];
sx q[10];
cx q[10],q[17];
rx(pi*0.005) q[10];
ry(pi*0.5) q[17];
cx q[17],q[10];
sxdg q[17];
s q[17];
cx q[10],q[17];
u3(pi*0.5,pi*1.8412881496,0) q[10];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[17];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[11];
rz(pi*0.495) q[18];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[11];
u3(pi*0.5,0,pi*1.6637118504) q[18];
sx q[11];
cx q[11],q[18];
rx(pi*0.005) q[11];
ry(pi*0.5) q[18];
cx q[18],q[11];
sxdg q[18];
s q[18];
cx q[11],q[18];
u3(pi*0.5,pi*1.8412881496,0) q[11];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[18];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[12];
rz(pi*-0.495) q[19];
u3(pi*0.5,0,pi*0.8362881496) q[12];
u3(pi*0.5,0,pi*0.8362881496) q[19];
sx q[12];
cx q[12],q[19];
rx(pi*0.005) q[12];
ry(pi*0.5) q[19];
cx q[19],q[12];
sxdg q[19];
s q[19];
cx q[12],q[19];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[12];
u3(pi*0.5,pi*0.6587118504,pi*1.0) q[19];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[13];
rz(pi*0.495) q[20];
u3(pi*0.5,pi*1.0,pi*0.6637118504) q[13];
u3(pi*0.5,0,pi*1.6637118504) q[20];
sx q[13];
cx q[13],q[20];
rx(pi*0.005) q[13];
ry(pi*0.5) q[20];
cx q[20],q[13];
sxdg q[20];
s q[20];
cx q[13],q[20];
u3(pi*0.5,pi*1.8412881496,0) q[13];
u3(pi*0.5,pi*0.8412881496,pi*1.0) q[20];

rx(pi*1.01) q[0];
rx(pi*1.01) q[1];
rx(pi*1.01) q[2];
rx(pi*1.01) q[3];
rx(pi*1.01) q[4];
rx(pi*1.01) q[5];
rx(pi*1.01) q[6];
rx(pi*1.01) q[7];
rx(pi*1.01) q[8];
rx(pi*1.01) q[9];
rx(pi*1.01) q[10];
rx(pi*1.01) q[11];
rx(pi*1.01) q[12];
rx(pi*1.01) q[13];
rx(pi*1.01) q[14];
rx(pi*1.01) q[15];
rx(pi*1.01) q[16];
rx(pi*1.01) q[17];
rx(pi*1.01) q[18];
rx(pi*1.01) q[19];
rx(pi*1.01) q[20];