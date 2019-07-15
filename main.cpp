#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>

using namespace Eigen;
using namespace std;

#define IdMat MatrixXd::Identity(data_size, data_size)

size_t length = 4;
size_t data_size = length * length * length;

enum Preconditioner
{
    NONE,
    JACOBI,
    NEUMANN,
    SPAI,
    SPAI_NO_DROP,
    SSOR,
    IP
};

enum Mode
{
    PCG,
    IMAGE,
    CONDITION
};

int Cube2Array(int idX, int idY, int idZ)
{
    return (idX < 0 ? idX + length : idX % length) +
           length * (idY < 0 ? idY + length : idY % length) +
           length * length * (idZ < 0 ? idZ + length : idZ % length);
}

double EnergyNorm(MatrixXd &A, VectorXd &x)
{
    return x.transpose() * A * x;
}

void PoissonPD(MatrixXd &A)
{ // Periodic Dirichlet
    A = 6 * IdMat;
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length; j++)
        {
            for (int k = 0; k < length; k++)
            {
                A(Cube2Array(i, j, k), Cube2Array(i - 1, j, k)) += -1;
                A(Cube2Array(i, j, k), Cube2Array(i + 1, j, k)) += -1;
                A(Cube2Array(i, j, k), Cube2Array(i, j - 1, k)) += -1;
                A(Cube2Array(i, j, k), Cube2Array(i, j + 1, k)) += -1;
                A(Cube2Array(i, j, k), Cube2Array(i, j, k - 1)) += -1;
                A(Cube2Array(i, j, k), Cube2Array(i, j, k + 1)) += -1;
            }
        }
    }
}

void PoissonPD(MatrixXd &A, int * pds)
{ // Dirichlet, periodic in Z
    A = 6 * IdMat;
    for (int iz = 0; iz < length; iz++)
    {
        for (int iy = 0; iy < length; iy++)
        {
            for (int ix = 0; ix < length; ix++)
            {
                A(Cube2Array(ix, iy, iz), Cube2Array(ix - 1, iy, iz)) += (ix == 0 && !pds[0]) ? 0 : -1;
                A(Cube2Array(ix, iy, iz), Cube2Array(ix + 1, iy, iz)) += (ix == length-1 && !pds[0]) ? 0 : -1;
                A(Cube2Array(ix, iy, iz), Cube2Array(ix, iy - 1, iz)) += (iy == 0 && !pds[1]) ? 0 : -1;
                A(Cube2Array(ix, iy, iz), Cube2Array(ix, iy + 1, iz)) += (iy == length-1 && !pds[1]) ? 0 : -1;
                A(Cube2Array(ix, iy, iz), Cube2Array(ix, iy, iz - 1)) += (iz == 0 && !pds[2]) ? 0 : -1;
                A(Cube2Array(ix, iy, iz), Cube2Array(ix, iy, iz + 1)) += (iz == length-1 && !pds[2]) ? 0 : -1;
            }
        }
    }
}

void RandomSym(MatrixXd &A)
{
    A = MatrixXd::Random(data_size, data_size);
    A = A * A.transpose();
}

void PoissonHD(MatrixXd &A)
{ // Homogenious Dirichlet
    A = 6 * IdMat;
    for (int n = 0; n < data_size; n++)
    {
        if (n - 1 < data_size) {
            A(n, n - 1) = -1;
            //adj[n].push_back(n-1);
        }
        if (n - length < data_size) {
            A(n, n - length) = -1;
            //adj[n].push_back(n - length);
        }
        if (n - (length * length) < data_size) {
            A(n, n - (length * length)) = -1;
            //adj[n].push_back(n - (length * length));
        }
        if (n + 1 < data_size) {
            A(n, n + 1) = -1;
            //adj[n].push_back(n + 1);
        }
        if (n + length < data_size) {
            A(n, n + length) = -1;
            //adj[n].push_back(n + length);
        }
        if (n + (length * length) < data_size) {
            A(n, n + (length * length)) = -1;
            //adj[n].push_back(n + (length * length));
        }
    }
}

void ColsOfA(int n, std::vector<int> & cols) {/*
    if (n - 1 < data_size)
        A(n, n - 1) = -1;
    if (n - length < data_size)
        A(n, n - length) = -1;
    if (n - (length * length) < data_size)
        A(n, n - (length * length)) = -1;
    if (n + 1 < data_size)
        A(n, n + 1) = -1;
    if (n + length < data_size)
        A(n, n + length) = -1;
    if (n + (length * length) < data_size)
        A(n, n + (length * length)) = -1;*/
}

void MatToImg(MatrixXd m, string fileName) {
    std::ofstream img(fileName);

    std::vector<int> cnt;
    bool flag = false;

    img << "P1" << endl;
    img << data_size << " " << data_size << endl;

    bool record;
    for (int i = 0; i < data_size; i++) {

        record = (int)(data_size/2) == i ? true : false;

        for (int j = 0; j < data_size; j++) {
            if (m(i,j) != 0) {
                img << 1;
                if (record) {
                    if (!flag) {
                        flag = true;
                        cnt.emplace_back(1);
                    } else {
                        cnt.back()++;
                    }
                }
            } else {
                img << 0;
                if (record) {
                    if (flag) {
                        flag = false;
                    }
                }
            }
            if(j != data_size-1) img << " ";
        }
        if(i != data_size-1) img << endl;
    }

    std::cout << endl;
    for (int const & elem : cnt) {
        std::cout << elem << " ";
    }
    std::cout << endl;
    img.close();
}

int main(int argc, char *argv[])
{
    Eigen::setNbThreads(8);

    // CG parameters //
    bool split = false;
    bool flex = false; //Doesnt do anything for neumann
    int iterations = 200;
    Preconditioner pc = NONE;
    Mode mode = PCG;

    if (argc > 1)
    {
        if (std::atoi(argv[1]) == 1)
        {
            mode = IMAGE;
        }
        else if (std::atoi(argv[1]) == 2)
        {
            mode = CONDITION;
        }
        else
        {
            mode = PCG;
        }
    }

    if (argc > 2)
    {
        if (std::atoi(argv[2]) == 1)
        {
            pc = JACOBI;
        }
        else if (std::atoi(argv[2]) == 2)
        {
            pc = NEUMANN;
        }
        else if (std::atoi(argv[2]) == 3)
        {
            pc = SPAI_NO_DROP;
        }
        else if (std::atoi(argv[2]) == 4)
        {
            pc = SPAI;
        }
        else if (std::atoi(argv[2]) == 5)
        {
            pc = SSOR;
        }
        else if (std::atoi(argv[2]) == 6)
        {
            pc = IP;
        }
        else
        {
            pc = NONE;
        }
    }

    if (argc > 3)
    {
        length = (std::atoi(argv[3]) > 0 ? std::atoi(argv[3]) : 4);
        data_size = length * length * length;
    }

    if (argc > 4)
    {
        iterations = (std::atoi(argv[4]) > 0 ? std::atoi(argv[4]) : 2000);
    }
    
    MatrixXd A;
    //MatrixXd P = MatrixXd::Zero(data_size,data_size);
    //std::vector<int> Ad[data_size];
    int periods[3] = {1,1,1}; // 0 0 1, 0 1 1, 1 1 1 are symetric, MI is only symetric for 1 1 1
    PoissonPD(A, periods);
    
    //JacobiSVD<MatrixXd> svd(A);
    //double kA = svd.singularValues().maxCoeff() / svd.singularValues().minCoeff();

    // cout << endl << A << endl; return 0;
    
    /*
    if (data_size == 216) {
            VectorXd perm = VectorXd::Zero(data_size);
            perm << 1,	7,	6,	2,	37,	36,	31,	32,	13,	12,	8,	43,	5,	42,	3,	38,	73,	35,	30,	72,	25,	67,	33,	26,	68,	19,	18,	14,	49,	11,	48,	9,	44,	79,	4,	41,	78,	39,	74,	109,	34,	29,	71,	24,
            66,	108,	61,	103,	27,	69,	20,	62,	104,	55,	17,	54,	15,	50,	85,	10,	47,	84,	45,	80,	115,	40,	77,	114,	75,	110,	145,	28,	70,	23,	65,	107,	60,	102,	144,	97,
            139, 21,	63,	105,	56,	98,	140,	91,	16,	53,	90,	51,	86,	121,	46,	83,	120,	81,	116,	151,	76,	113	,150,	111,	146	,181,	22,	64,	106,	59,	101,	143	,96	,138,
            180,	133,	175,	57,	99,	141,	92,	134,	176,	127	,52	,89	,126,	87,	122	,157,	82,	119	,156,	117	,152,	187	,112,	149	,186,	147	,182,	58	,100,	142,	95,
            137,	179,	132,	174,	216,	169,	211,	93,	135,	177,	128,	170	,212,	163	,88	,125,	162,	123,	158	,193	,118,	155,	192	,153,	188	,148,	185	,
            183,	94,	136	,178,	131,	173,	215	,168,	210	,205,	129	,171	,213,	164,	206,	199	,124,	161	,198,	159	,194,	154,	191,189,	184	,130,	172	,214,	167,
            209	,204,	165,	207,	200,	160,	197	,195,	190,	166	,208,	203	,201,	196,	202;
            cout << endl << "P loaded" << endl;

            for (int i = 0; i < data_size; i++) {
                P(perm(i)-1,i) = 1;
            }
        }
        */

    //MatrixXd P = MatrixXd::Zero(data_size,data_size);
    //CuthillMcKee(P, Ad);

    //A = P*A;

    //MatToImg(test, "reorder");
    //return 0;

    VectorXd b = VectorXd(data_size);
    VectorXd start = VectorXd(data_size);
    for (size_t i = 0; i < data_size; i++)
        start(i) = (double)((i * 5) % 11) - 5.0;
    b = A * start;

    // Preconditioner setup //
    MatrixXd MI, L, U;
    MatrixXd A_L = A.triangularView<StrictlyLower>();
    MatrixXd A_II = A.diagonal().cwiseInverse().asDiagonal();
    MatrixXd A_BII = MatrixXd::Identity(data_size, data_size);
    for (int i = 0; i < data_size-1; i++) {
        Matrix2d el;
        if (A(i,i+1) != 0) {
            el = A.block(i,i,2,2);
            el(1,0) = 1;
            el(0,1) = 1;
            el *= 1.0 / (6*6 - (-1)*(-1));
            
            A_BII.block(i,i,2,2) = el;
        }
    }

    stringstream name;
    switch (pc)
    {
    case NONE:
    {
        name << "no preconditioner";
        //MI = IdMat;
        break;
    }
    case JACOBI:
    {
        name << "JACOBI";
        MI = A_BII;
        break;
    }
    case NEUMANN:
    { // Improved by block diagonal
        name << "NEUMANN";
        int s = 4;
        double w = 0.9;
        MatrixXd N = IdMat - w * A_BII * A;
        MI = IdMat;
        MatrixXd tmp = IdMat;
        for (int i = 1; i < s; i++)
        {
            tmp *= N;
            MI += tmp;
        }
        MI *= A_BII;
        break;
    }
    case SPAI_NO_DROP:
    {
        name << "SPAI_NO_DROP";
        double s = (A * A).trace() / (A * A * (A * A).transpose()).trace();
        int k = 3;
        MI = s*A;
        for (int i = 0; i < k; i++)
        {
            MatrixXd G = IdMat - A * MI;
            double frob = (A * G).norm();
            double alpha = (G.transpose() * A * G).trace() / (frob * frob);
            MI += alpha * G;
        }
        break;
    }
    case SPAI:
    {
        name << "SPAI";
        int n0 = 1;
        int ni = 100;
        MI = A_BII;
        MatrixXd Pattern = A;

        int test = 0;
        
        for (int outer = 0; outer < n0; outer++) {
        //    
            //#pragma omp parallel for
            for (int j = 0; j < data_size; j++) { // Minimizing columns
                //if(j==test) cout << "m0: " << MI.col(j).transpose() << endl;
                VectorXd r = VectorXd::Unit(data_size,j) - (A*MI.col(j)); // (1) r has size 7
                //if(j==test) cout << "r: " << r.transpose() << endl;
                //for (int inner = 0; inner < ni; inner++) {
                    //VectorXd t = MC*r;
                    VectorXd d = r;
                    for (int i = 0; i < data_size; i++) { // Set sparsity of search direction (2)
                        if (Pattern(i,j) == 0.0) {
                            d(i) = 0.0;
                        }
                    }
                    //if (j==test) cout << "d: " << d.transpose() << endl;
                    VectorXd q = A*d; // (3) q has size 7 => Need 7x7x7 cube
                    //if (j==test) cout << "q: " << q.transpose() << endl;
                    double a = r.dot(q) / q.dot(q); // (4)
                    //if(j==test) cout << "rq: " << r.dot(q) << " qq: " << q.dot(q) << endl;
                    MI.col(j) += a*d; // (5)
                    //if(j==test) cout << "m1: " << MI.col(j).transpose() << endl;
                    //if (j==test) return 0;
                //}
            }
        }
        break;
    }
    case SSOR:
    {
        name << "SSOR";
        double w = 0.9;
        MI = (2 - w) * (IdMat - (A_L * (w * A_BII))).transpose() * (w * A_BII) * (IdMat - (A_L * (w * A_BII)));
        break;
    }
    case IP: {
        name << "IP";
        MI = (IdMat-A_L*A_II)*(IdMat-A_L*A_II).transpose();
        break;
    }
    }
    std::cout << name.str();

    // Exporting the sparsity pattern of MI //
    if (mode == IMAGE) {
        
        name << ".ppm";
        MatToImg(MI, name.str());
        
        return 0;
    }

    //out << endl << MI.col(0).transpose() - MI.row(0) << endl; return 0;

    MatrixXd shape = MatrixXd::Zero(data_size,data_size);
    for (int i = 0; i < data_size; i++) {
        for (int j = 0; j < data_size; j++) {
            if (MI(i,j) != 0.0) {
                shape(i,j) = 1.0;
            }
        }
    }

    cout << MI << endl;
    return 0;

    if (mode == CONDITION) {
        MatrixXd Res = MI*A;
        JacobiSVD<MatrixXd> svdRes(Res);
        double kRes = svdRes.singularValues().maxCoeff() / svdRes.singularValues().minCoeff();
        JacobiSVD<MatrixXd> svdA(A);
        double kA = svdA.singularValues().maxCoeff() / svdA.singularValues().minCoeff();

        cout << endl << "Cond(A): " << kA << ", Cond(M*A): " << kRes << endl; return 0;
    }

    /*if (analysis)
    {
        
        VectorXd nonZero = VectorXd::Zero(data_size);
        for (int i = 0; i < data_size; i++)
        {
            for (int j = 0; j < data_size; j++)
            {
                if (MI(i, j) != 0)
                    nonZero(i)++;
            }
        }

        
        int iters = ceil((log10(1e-4 / EnergyNorm(A, r0)) / (log10((sqrt(kA) - 1) / (sqrt(kA) + 1)))/3.172));

        double sqrtP = 4;
        double P = sqrtP*sqrtP;
        double nzr = (double)nonZero.maxCoeff();

        //cout << endl << "Non zeros per entry:" << endl << nonZero;
        cout << endl
             << "Max non-Zeros per row: " << nonZero.maxCoeff() << std::endl;
        cout << endl
             << "Iterations Necessary: " << iters << std::endl;
        cout << endl
             << "Maximum cost per node: " <<
              + // preconditioner
             iters * (((nzr + 7)*(data_size/sqrtP)) + (2*(sqrtP-1)*sqrtP)) //cost of A*x and C*x
             << std::endl;
        return 0; 
    }*/
    std::cout << "Iters: " << iterations << ", Length: " << length << ", PC: ";
    std::cout << (split ? ", Split" : "") << endl;

    VectorXd x = VectorXd::Zero(data_size);
    VectorXd r = b;
    VectorXd z, p, rh, r_old, rh_old;
    double alpha, beta, rz_new, rz_old;
    int recomp = 10;

    if (pc == NONE) {
        z = r;
    } else {
        z = MI * r;
    }
    p = z;

    rz_new = r.dot(z);

    std::cout << "Iteration, Error" << endl;

    //while (true) {
        for (int j = 0; j < iterations; j++)
        {
            alpha = rz_new / (A * p).dot(p);
            x = x + (alpha * p);

            r_old = r;
            r = r - alpha * A * p;
            if (pc == NONE) {
                z = r;
            } else {
                z = MI * r;
            }

            rz_old = rz_new;
            
            rz_new = r.dot(z);

            if (flex)
            {
                beta = z.dot(r - r_old) / rz_old;
            }
            else
            {
                beta = rz_new / rz_old;
            }

            p = z + beta * p;

            std::cout << j + 1 << ", " << (split ? rh.norm() : r.norm()) << endl;
        }

        /*cout << "Refresh" << endl;

        r = b - A*x;
        if (pc == NONE) {
            z = r;
        } else {
            z = MI * r;
        }
        p = z;

        rz_new = r.dot(z);
    }*/

    cout << "Final rate in A norm: " << log10(sqrt(r.transpose() * A * r)) / iterations << endl;

    return 0;
}
