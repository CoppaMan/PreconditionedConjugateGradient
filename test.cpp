oid PoissonSolverPCG::ComputeMR(size_t iterations) {
  const int kDensity = 1;
  const int kSize = kDensity*2 + 1;
  const int gCenter = kDensity + 2; // Center of array cube
  const int gSize = gCenter*2 + 1; // Size of the array cube

  // Create MR grid

  struct MRKernel {
    std::vector<BlockInfo> & precond;
    int kernelSize;
    const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
    const StencilInfo stencil = StencilInfo(-1,-1,-1,2,2,2,false, 6, 1,2,3,5,6,7); // TODO: 

    PrecondKernel(std::vector<BlockInfo> & precondInfo, BlockVar result, BlockVar vector) : precond(precondInfo), res(result), vec(vector) {}

    void operator()(PCGLab & l, const BlockInfo& info, PCGBlock& o) const
    {
      PrecondBlock & p = *((PrecondBlock *) precond.at(info.blockID).ptrBlock);
      for(int iz=0; iz<PCGBlock::sizeZ; ++iz)
      for(int iy=0; iy<PCGBlock::sizeY; ++iy)
      for(int ix=0; ix<PCGBlock::sizeX; ++ix) {
        size_t count = 0;
        Real sum = 0;
        for(int rz=0; rz < kernelSize; ++rz) 
        for(int ry=0; ry < kernelSize; ++ry)
        for(int rx=0; rx < kernelSize; ++rx) {
          if (std::abs(rx-1) + std::abs(ry-1) + std::abs(rz-1) <= 1) { //Todo get entries of column to form row in kernel of precond grid.
      //if (info.blockID == 0 && ix==0 && iy==0 && iz==0) std::cout << "(" << rx << "," << ry << "," << rz << ") p:" << p(ix,iy,iz).kernel[count] << std::endl;
            sum += p(ix,iy,iz).kernel[count++] * (&(l(ix+(rx-1), iy+(ry-1), iz+(rz-1)).x))[vec];
          }
        }
        (&(o(ix,iy,iz).x))[res] = sum;
      }
    }
  };

  // Column kernel (all cells touching our current element)
  for(size_t i=0; i<precondInfo->size(); i++) {
    assert((size_t) precondInfo->at(i).blockID == i);
    BlockInfo & info = precondInfo->at(i);
    PrecondBlock & pb = *((PrecondBlock *) info.ptrBlock);
    for(int iz=0; iz<PrecondBlock::sizeZ; ++iz)
    for(int iy=0; iy<PrecondBlock::sizeY; ++iy)
    for(int ix=0; ix<PrecondBlock::sizeX; ++ix) {
      
      // True if block touches non periodic boundary in given direction
      bool wWall = pcgPos[0] == 0 && info.index[0] == 0 && !periodic;
      bool eWall = pcgPos[0] == pcgSize[0]-1 && info.index[0] == (int)mybpd[0]-1 && !periodic;
      bool sWall = pcgPos[1] == 0 && info.index[1] == 0 && !periodic;
      bool nWall = pcgPos[1] == pcgSize[1]-1 && info.index[1] == (int)mybpd[1]-1 && !periodic;
      bool fWall = pcgPos[2] == 0 && info.index[2] == 0 && !periodic;
      bool bWall = pcgPos[2] == pcgSize[2]-1 && info.index[2] == (int)mybpd[2]-1 && !periodic;

      Real m[gSize][gSize][gSize] = {};

      const Real det = 1.0 / 35.0;
      bool wStop = pcgPos[0] == 0 && info.index[0] == 0; // Jacobian is never periodic
      bool eStop = pcgPos[0] == pcgSize[0]-1 && info.index[0] == (int)mybpd[0]-1;
      m[gCenter][gCenter][gCenter] = det*6.0;
      if (!(ix+(kx-1) < 0 && wStop)) {
        m[gCenter-1][gCenter][gCenter] = det;
      }
      if (!(ix+(kx-1) > PrecondBlock::sizeX-1 && eStop)) {
        m[gCenter+1][gCenter][gCenter] = det;
      }

      /*if (ix == 0 && iy == 0 && iz == 0 && i == 0) {
        for(int pz = 0; pz < 5; pz++) {
          for (int py = 0; py < 5; py++) {
            for (int px = 0; px < 5; px++) {
              std::cout << m[px][py][pz] << " ";
            } std::cout << std::endl;
          } std::cout << std::endl << std::endl;
        }
      }*/

      Real r[gSize][gSize][gSize] = {};
      Real d[gSize][gSize][gSize] = {};
      Real q[gSize][gSize][gSize] = {};
    
      for (size_t itr = 0; itr < iterations; itr++) { // MR Iteration

        for(int rz=0; rz<gSize; ++rz) // r_j = e_j - A*m_j
        for(int ry=0; ry<gSize; ++ry)
        for(int rx=0; rx<gSize; ++rx) {
          bool cOut = (wWall && ix + rx-gCenter < 0) || (eWall && ix + rx-gCenter > PrecondBlock::sizeX-1) || // If current cell is outside border
            (sWall && iy + ry-gCenter < 0) || (nWall && iy + ry-gCenter > PrecondBlock::sizeY-1) ||
            (fWall && iz + rz-gCenter < 0) || (bWall && iz + rz-gCenter > PrecondBlock::sizeZ-1);
          bool wOut = wWall && ix + (rx-gCenter)-1 < 0;
          bool eOut = eWall && ix + (rx-gCenter)+1 > PrecondBlock::sizeX-1;
          bool sOut = sWall && iy + (ry-gCenter)-1 < 0;
          bool nOut = nWall && iy + (ry-gCenter)+1 > PrecondBlock::sizeY-1;
          bool fOut = fWall && iz + (rz-gCenter)-1 < 0;
          bool bOut = bWall && iz + (rz-gCenter)+1 > PrecondBlock::sizeZ-1;

          if (std::abs(rx-gCenter) + std::abs(ry-gCenter) + std::abs(rz-gCenter) <= kDensity+1) { // r is 1 larger than m
            if ( cOut ) { // Do not compute elements beyond walls
              r[rx][ry][rz] = 0;
            } else {
              Real sum = -6*m[rx][ry][rz]; // -A*m
              sum += wOut ? 0.0 : m[rx-1][ry][rz];
              sum += eOut ? 0.0 : m[rx+1][ry][rz];
              sum += sOut ? 0.0 : m[rx][ry-1][rz];
              sum += nOut ? 0.0 : m[rx][ry+1][rz];
              sum += fOut ? 0.0 : m[rx][ry][rz-1];
              sum += bOut ? 0.0 : m[rx][ry][rz+1];
              if (rx==gCenter && ry == gCenter && rz == gCenter) sum += 1; // +e_j
              r[rx][ry][rz] = sum;

              if (std::abs(rx-gCenter) + std::abs(ry-gCenter) + std::abs(rz-gCenter) <= kDensity) { // d is size of m
                d[rx][ry][rz] = sum;
              }
	          }
          }
        }
        /*if (ix == 0 && iy == 0 && iz == 0 && i == 0) {
          for(int pz = 0; pz < 7; pz++) {
            for (int py = 0; py < 7; py++) {
              for (int px = 0; px < 7; px++) {
                std::cout << d[px][py][pz] << "(" << r[px][py][pz] << ") ";
              } std::cout << std::endl;
            } std::cout << std::endl << std::endl;
          }
        }*/

        for(int qz=0; qz<gSize; ++qz) // q_j = A*d_j
        for(int qy=0; qy<gSize; ++qy)
        for(int qx=0; qx<gSize; ++qx) {
  
          bool cOut = (wWall && ix + qx-gCenter < 0) || (eWall && ix + qx-gCenter > PrecondBlock::sizeX-1) ||
            (sWall && iy + qy-gCenter < 0) || (nWall && iy + qy-gCenter > PrecondBlock::sizeY-1) ||
            (fWall && iz + qz-gCenter < 0) || (bWall && iz + qz-gCenter > PrecondBlock::sizeZ-1);
          bool wOut = wWall && ix + (qx-gCenter)-1 < 0;
          bool eOut = eWall && ix + (qx-gCenter)+1 > PrecondBlock::sizeX-1;
          bool sOut = sWall && iy + (qy-gCenter)-1 < 0;
          bool nOut = nWall && iy + (qy-gCenter)+1 > PrecondBlock::sizeY-1;
          bool fOut = fWall && iz + (qz-gCenter)-1 < 0;
          bool bOut = bWall && iz + (qz-gCenter)+1 > PrecondBlock::sizeZ-1;

          if (std::abs(qx-gCenter) + std::abs(qy-gCenter) + std::abs(qz-gCenter) <= kDensity+1) {
            if ( cOut ) {
              q[qx][qy][qz] = 0;
            } else {
              Real sum = 6*d[qx][qy][qz];
              sum -= wOut ? 0.0 : d[qx-1][qy][qz];
              sum -= eOut ? 0.0 : d[qx+1][qy][qz];
              sum -= sOut ? 0.0 : d[qx][qy-1][qz];
              sum -= nOut ? 0.0 : d[qx][qy+1][qz];
              sum -= fOut ? 0.0 : d[qx][qy][qz-1];
              sum -= bOut ? 0.0 : d[qx][qy][qz+1];
              q[qx][qy][qz] = sum;
            }
          }
        }

        /*if (ix == 0 && iy == 0 && iz == 0 && i == 0) {
          for(int pz = 0; pz < 7; pz++) {
            for (int py = 0; py < 7; py++) {
              for (int px = 0; px < 7; px++) {
                std::cout << q[px][py][pz] << " ";
              } std::cout << std::endl;
            } std::cout << std::endl << std::endl;
          }
	      }*/

        Real sum_rq = 0;
        Real sum_qq = 0;
        for(int rz=0; rz<gSize; ++rz) // dot products
        for(int ry=0; ry<gSize; ++ry)
        for(int rx=0; rx<gSize7; ++rx) {
          if (std::abs(rx-gCenter) + std::abs(ry-gCenter) + std::abs(rz-gCenter) <= kDensity+1) {
            sum_qq += q[rx][ry][rz]*q[rx][ry][rz];
	          sum_rq += r[rx][ry][rz]*q[rx][ry][rz];
          }
        }

        Real a = sum_rq / sum_qq;
	      //if (ix == 0 && iy == 0 && iz == 0 && i == 0)std::cout << std::endl << "r: " << sum_rq << " q: " << sum_qq << std::endl;

        for(int rz=0; rz<kSize; ++rz) // m_j = m_j + alpha*r_j 
        for(int ry=0; ry<kSize; ++ry)
        for(int rx=0; rx<kSize; ++rx) {
          if (std::abs(rx-kDensity) + std::abs(ry-kDensity) + std::abs(rz-kDensity) <= kDensity) {
            m[rx+2][ry+2][rz+2] += a*r[rx+2][ry+2][rz+2];
          }
        }

        /*if (ix == 0 && iy == 0 && iz == 0 && i == 0) {
          for(int pz = 0; pz < 7; pz++) {
            for (int py = 0; py < 7; py++) {
              for (int px = 0; px < 7; px++) {
                std::cout << m[px][py][pz] << " ";
              } std::cout << std::endl;
            } std::cout << std::endl << std::endl;
          }
        }*/
      }

      /*size_t count = 0;
      for(int rz=0; rz<kSize; ++rz) // Store in PrecondElement
      for(int ry=0; ry<kSize; ++ry)
      for(int rx=0; rx<kSize; ++rx) {
        if (std::abs(rx-kDensity) + std::abs(ry-kDensity) + std::abs(rz-kDensity) <= kDensity) {
	        pb(ix,iy,iz).kernel[count++] = m[rx+2][ry+2][rz+2];
        }
      }
      */

      // Loads elements of MR columns into the correct kernel spots
      Operator::compute<MRKernel, MRGridMPI, MRBlock, MRLabMPI>(MR, *MRGrid);
    }
  }

  // Destroy MR grid
}


struct MRElement
{
  typedef Real RealType;
  // Column associated with linear order of this element
  Real col[7];
  void clear() {}
  MRElement(const MRElement& c) = delete;
};

using FluidBlock = BaseBlock<FluidElement>;
using FluidGrid    = cubism::Grid<FluidBlock, aligned_allocator>;
using FluidGridMPI = cubism::GridMPI<FluidGrid>;

using PCGBlock   = BaseBlock<PCGElement>;
using PCGGrid    = cubism::Grid<PCGBlock, aligned_allocator>;
using PCGGridMPI = cubism::GridMPI<PCGGrid>;

using MRBlock   = BaseBlock<MRElement>;
using MRGrid    = cubism::Grid<MRBlock, aligned_allocator>;
using MRGridMPI = cubism::GridMPI<MRGrid>;

using PrecondBlock   = BaseBlock<PrecondElement>;
using PrecondGrid    = cubism::Grid<PrecondBlock, aligned_allocator>;
using PrecondGridMPI = cubism::GridMPI<PrecondGrid>;

using PenalizationBlock   = BaseBlock<PenalizationHelperElement>;
using PenalizationGrid    = cubism::Grid<PenalizationBlock, aligned_allocator>;
using PenalizationGridMPI = cubism::GridMPI<PenalizationGrid>;

using Lab          = BlockLabBC<FluidBlock, aligned_allocator>;
using LabMPI       = cubism::BlockLabMPI<Lab>;

using PCGLab = cubism::BlockLab<PCGBlock, aligned_allocator>;
using PCGLabMPI = cubism::BlockLabMPI<PCGLab>;

using MRLab = cubism::BlockLab<MRBlock, aligned_allocator>;
using MRLabMPI = cubism::BlockLabMPI<MRLab>;
