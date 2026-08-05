#ifndef THERMAL_INFLATION_FIELD_SNAPSHOT_HPP
#define THERMAL_INFLATION_FIELD_SNAPSHOT_HPP

/* Lightweight 3D field snapshot writer for thermal-inflation CosmoLattice runs.
 *
 * Writes program-unit fldS/piS values to field_states/snapshot_{step:010d}.raw plus a
 * manifest.csv for post-run conversion to numba-compatible NPZ (see
 * tools/export_cl_snapshots.py).
 *
 * Formats:
 *   FLPI (0x464C5049): single scalar phi + pi
 *   FLP2 (0x464C5032): two components phi1, phi2, pi1, pi2 (complex field)
 *
 * MPI: every rank packs its local interior into a full N^3 buffer (zeros elsewhere)
 * and MPI_Reduce(SUM) gathers to root, which writes the file. Without this, only
 * rank 0's slab was saved (7/8 of the volume looked "dead").
 */

#include <sstream>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "TempLat/parallel/mpi/mpitypeconstants.h"

#ifndef NOMPI
#include <mpi.h>
#endif

namespace ThermalInflation {

using TempLat::operator"" _c;

constexpr uint32_t SNAPSHOT_MAGIC = 0x464C5048u;     // 'FLPH' phi only (legacy)
constexpr uint32_t SNAPSHOT_MAGIC_PI = 0x464C5049u;  // 'FLPI' phi + pi
constexpr uint32_t SNAPSHOT_MAGIC_PI2 = 0x464C5032u; // 'FLP2' phi1, phi2, pi1, pi2

#pragma pack(push, 1)
struct SnapshotHeader {
    uint32_t magic;
    uint32_t N;
    int64_t step;
    double t;
    double T;
    double a;
    double H;
    double fStar;
};

struct SnapshotHeader2 {
    SnapshotHeader base;
    uint32_t nScalars;
};
#pragma pack(pop)

inline bool ensureDir(const std::string& path) {
    struct stat st {};
    if (stat(path.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    return mkdir(path.c_str(), 0755) == 0;
}

template <class Model>
inline double snapshotHubble(const Model& model) {
    return model.prescribedHubble();
}

class FieldSnapshotWriter {
public:
    FieldSnapshotWriter() = default;

    void configure(const std::string& outputDir, bool enabled,
                   int coarseSteps, int denseSteps, double phiThresholdGeV,
                   double fStar, int latticeN) {
        enabled_ = enabled;
        latticeN_ = latticeN;
        fStar_ = fStar;
        coarseStepFreq_ = std::max(1, coarseSteps);
        denseStepFreq_ = std::max(1, denseSteps > 0 ? denseSteps : coarseSteps);
        denseEnabled_ = enabled && denseSteps > 0 && phiThresholdGeV > 0.0;
        phiThresholdGeV_ = phiThresholdGeV;
        denseActive_ = false;
        stepFreq_ = coarseStepFreq_;

        if (!enabled_) return;

        dir_ = outputDir;
        if (!dir_.empty() && dir_.back() != '/') dir_ += '/';
        stateDir_ = dir_ + "field_states/";
        ensureDir(stateDir_);
        openManifest();
    }

    template <class Model>
    void maybeSave(Model& model, int n, double t) {
        if (!enabled_) return;

        updateStepFreq(model);

        if (stepFreq_ <= 0 || (n % stepFreq_) != 0) return;

        // All ranks must enter writeSnapshot (MPI_Reduce collective).
        writeSnapshot(model, n, t);
    }

private:
    template <class Model>
    double maxPhiGeV(Model& model) const {
        const int nComp = model.activeScalars();
        double phiMaxGeV = 0.0;
        auto toolbox = model.getToolBox();
        auto& it = toolbox->itX();
        for (it.begin(); it.end(); ++it) {
            const double p1 = std::abs(static_cast<double>(model.fldS(0_c).get(it()))) * fStar_;
            if (nComp < 2) {
                if (p1 > phiMaxGeV) phiMaxGeV = p1;
            } else {
                const double p2 = std::abs(static_cast<double>(model.fldS(1_c).get(it()))) * fStar_;
                const double rho = std::sqrt(p1 * p1 + p2 * p2);
                if (rho > phiMaxGeV) phiMaxGeV = rho;
            }
        }
#ifndef NOMPI
        MPI_Allreduce(MPI_IN_PLACE, &phiMaxGeV, 1, MPI_DOUBLE, MPI_MAX,
                      toolbox->mGroup.getBaseComm());
#endif
        return phiMaxGeV;
    }

    template <class Model>
    void updateStepFreq(Model& model) {
        if (denseActive_) {
            stepFreq_ = denseStepFreq_;
            return;
        }
        stepFreq_ = coarseStepFreq_;

        if (!denseEnabled_) return;

        const double phiMaxGeV = maxPhiGeV(model);
        if (phiMaxGeV > phiThresholdGeV_) {
            denseActive_ = true;
            stepFreq_ = denseStepFreq_;
            if (model.getToolBox()->amIRoot()) {
                std::cout << "\n*** phi threshold exceeded: max|phi|="
                          << phiMaxGeV << " > " << phiThresholdGeV_
                          << " (GeV)\n*** Switching to dense snapshots: every "
                          << denseStepFreq_ << " steps\n\n";
            }
        }
    }

    /** Convert CosmoLattice signed spatial coord to [0, N). */
    static size_t toIndex0N(ptrdiff_t c, int N) {
        return static_cast<size_t>(c >= 0 ? c : c + N);
    }

    /** Pack local interior sites into a full N^3 buffer (zeros elsewhere).
     *
     * IMPORTANT: use it.getVec() (signed global coords). Do NOT pass it()
     * (memory offset) into getCoordConfiguration0N — that API expects a loop
     * index into the coordinate cache, not an offset.
     */
    template <class Model, class Getter>
    void packGlobalField(Model& model, std::vector<float>& buf, Getter getter) const {
        const size_t n3 = buf.size();
        std::fill(buf.begin(), buf.end(), 0.0f);
        auto toolbox = model.getToolBox();
        const int N = latticeN_;
        auto& it = toolbox->itX();
        for (it.begin(); it.end(); ++it) {
            const auto c = it.getVec();  // signed global coordinates
            if (c.size() < 3) continue;
            const size_t ix = toIndex0N(c[0], N);
            const size_t iy = toIndex0N(c[1], N);
            const size_t iz = toIndex0N(c[2], N);
            if (ix >= static_cast<size_t>(N) || iy >= static_cast<size_t>(N) ||
                iz >= static_cast<size_t>(N)) {
                continue;
            }
            const size_t gidx = (ix * static_cast<size_t>(N) + iy) * static_cast<size_t>(N) + iz;
            if (gidx >= n3) continue;
            buf[gidx] = static_cast<float>(getter(it()));
        }
    }

    template <class Model>
    void reduceToRoot(Model& model, std::vector<float>& buf) const {
#ifndef NOMPI
        auto toolbox = model.getToolBox();
        const int nprocs = toolbox->getNProcesses();
        if (nprocs <= 1) return;
        MPI_Comm comm = toolbox->mGroup.getBaseComm();
        if (toolbox->amIRoot()) {
            MPI_Reduce(MPI_IN_PLACE, buf.data(), static_cast<int>(buf.size()),
                       MPI_FLOAT, MPI_SUM, 0, comm);
        } else {
            MPI_Reduce(buf.data(), nullptr, static_cast<int>(buf.size()),
                       MPI_FLOAT, MPI_SUM, 0, comm);
        }
#else
        (void)model;
        (void)buf;
#endif
    }

    template <class Model>
    void writeSnapshot(Model& model, int n, double t) {
        auto toolbox = model.getToolBox();
        const bool isRoot = toolbox->amIRoot();
        const int64_t step = n;
        const int nComp = model.activeScalars();
        const double T = model.currentT();
        const double a = model.aI;
        const double H = snapshotHubble(model);
        const int stageId = model.expansionStageId();
        const double rhoM = model.rhoMatter();

        const size_t n3 = static_cast<size_t>(latticeN_) *
                          static_cast<size_t>(latticeN_) *
                          static_cast<size_t>(latticeN_);

        std::vector<float> buf0(n3);
        std::vector<float> buf1;
        std::vector<float> pi0(n3);
        std::vector<float> pi1;
        if (nComp > 1) {
            buf1.resize(n3);
            pi1.resize(n3);
        }

        packGlobalField(model, buf0, [&](auto idx) {
            return model.fldS(0_c).get(idx);
        });
        reduceToRoot(model, buf0);

        packGlobalField(model, pi0, [&](auto idx) {
            return model.piS(0_c).get(idx);
        });
        reduceToRoot(model, pi0);

        if (nComp > 1) {
            packGlobalField(model, buf1, [&](auto idx) {
                return model.fldS(1_c).get(idx);
            });
            reduceToRoot(model, buf1);

            packGlobalField(model, pi1, [&](auto idx) {
                return model.piS(1_c).get(idx);
            });
            reduceToRoot(model, pi1);
        }

        if (!isRoot) return;

        const std::string fname = "snapshot_" + zeroPad(step, 10) + ".raw";
        const std::string fpath = stateDir_ + fname;

        SnapshotHeader hdr{};
        hdr.magic = (nComp > 1) ? SNAPSHOT_MAGIC_PI2 : SNAPSHOT_MAGIC_PI;
        hdr.N = static_cast<uint32_t>(latticeN_);
        hdr.step = step;
        hdr.t = t;
        hdr.T = T;
        hdr.a = a;
        hdr.H = H;
        hdr.fStar = fStar_;

        std::ofstream out(fpath, std::ios::binary);
        if (!out) return;
        if (nComp > 1) {
            SnapshotHeader2 hdr2{};
            hdr2.base = hdr;
            hdr2.nScalars = static_cast<uint32_t>(nComp);
            out.write(reinterpret_cast<const char*>(&hdr2), sizeof(hdr2));
        } else {
            out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        }
        out.write(reinterpret_cast<const char*>(buf0.data()),
                  static_cast<std::streamsize>(buf0.size() * sizeof(float)));
        if (nComp > 1) {
            out.write(reinterpret_cast<const char*>(buf1.data()),
                      static_cast<std::streamsize>(buf1.size() * sizeof(float)));
        }
        out.write(reinterpret_cast<const char*>(pi0.data()),
                  static_cast<std::streamsize>(pi0.size() * sizeof(float)));
        if (nComp > 1) {
            out.write(reinterpret_cast<const char*>(pi1.data()),
                      static_cast<std::streamsize>(pi1.size() * sizeof(float)));
        }
        out.close();

        if (manifest_.is_open()) {
            manifest_ << step << ','
                      << std::setprecision(16) << t << ','
                      << T << ',' << a << ',' << H << ','
                      << fStar_ << ',' << nComp << ','
                      << stageId << ',' << rhoM << ','
                      << fname << '\n';
            manifest_.flush();
        }
    }

    static std::string zeroPad(int64_t v, int width) {
        std::ostringstream oss;
        oss << std::setw(width) << std::setfill('0') << v;
        return oss.str();
    }

    void openManifest() {
        const std::string mpath = stateDir_ + "manifest.csv";
        const bool exists = std::ifstream(mpath).good();
        manifest_.open(mpath, std::ios::out | std::ios::app);
        if (!exists && manifest_.is_open()) {
            manifest_ << "step,t,T,a,H,fStar,n_scalars,expansion_stage,rho_m,filename\n";
        }
    }

    bool enabled_ = false;
    bool denseEnabled_ = false;
    bool denseActive_ = false;
    int latticeN_ = 0;
    int coarseStepFreq_ = 1;
    int denseStepFreq_ = 1;
    int stepFreq_ = 1;
    double phiThresholdGeV_ = 0.0;
    double fStar_ = 1.0;
    std::string dir_;
    std::string stateDir_;
    std::ofstream manifest_;
};

}  // namespace ThermalInflation

#endif  // THERMAL_INFLATION_FIELD_SNAPSHOT_HPP
