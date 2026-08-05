#ifndef THERMAL_INFLATION_FIELD_SNAPSHOT_V2_HPP
#define THERMAL_INFLATION_FIELD_SNAPSHOT_V2_HPP

/* 3D field snapshot writer for CosmoLattice v2 thermal-inflation runs.
 *
 * Same on-disk format as v1 (field_states/snapshot_*.raw + manifest.csv) so
 * tools/export_cl_snapshots.py keeps working. TempLat v2 removed itX()/get(i),
 * so packing uses Field::getLocalNDHostView() (pulls the host mirror) plus the
 * config-space localStarts for global indexing.
 *
 * Supports --steps / --phi_threshold / --steps_dense: coarse cadence until
 * max|phi| (GeV) crosses the threshold, then dense cadence.
 */

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifndef HAVE_MPI
// Serial builds: TempLat stubs MPI_Comm as int; no real MPI calls.
#else
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
        writeSnapshot(model, n, t);  // all ranks: MPI_Reduce collective
    }

private:
    /** Pack one field's local interior into a full N^3 buffer (zeros elsewhere). */
    template <class Model, class FieldT>
    void packField(Model& model, FieldT&& field, std::vector<float>& buf) const {
        std::fill(buf.begin(), buf.end(), 0.0f);
        auto toolbox = model.getToolBox();
        const auto& layout = toolbox->mLayouts.getConfigSpaceLayout();
        const auto& starts = layout.getLocalStarts();
        const auto& localSizes = layout.getLocalSizes();
        const int N = latticeN_;

        // Host subview is the local interior (ghosts stripped).
        auto host = field.getLocalNDHostView();

        for (size_t i = 0; i < static_cast<size_t>(localSizes[0]); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(localSizes[1]); ++j) {
                for (size_t k = 0; k < static_cast<size_t>(localSizes[2]); ++k) {
                    const size_t ix = static_cast<size_t>(starts[0]) + i;
                    const size_t iy = static_cast<size_t>(starts[1]) + j;
                    const size_t iz = static_cast<size_t>(starts[2]) + k;
                    if (ix >= static_cast<size_t>(N) || iy >= static_cast<size_t>(N) ||
                        iz >= static_cast<size_t>(N)) {
                        continue;
                    }
                    const size_t gidx =
                        (ix * static_cast<size_t>(N) + iy) * static_cast<size_t>(N) + iz;
                    if (gidx >= buf.size()) continue;
                    buf[gidx] = static_cast<float>(host(i, j, k));
                }
            }
        }
    }

    template <class Model>
    double maxPhiGeV(Model& model) const {
        const int nComp = model.activeScalars();
        auto toolbox = model.getToolBox();
        const auto& layout = toolbox->mLayouts.getConfigSpaceLayout();
        const auto& localSizes = layout.getLocalSizes();

        auto host0 = model.fldS(0_c).getLocalNDHostView();
        double phiMaxGeV = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(localSizes[0]); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(localSizes[1]); ++j) {
                for (size_t k = 0; k < static_cast<size_t>(localSizes[2]); ++k) {
                    const double p1 = std::abs(static_cast<double>(host0(i, j, k))) * fStar_;
                    if (nComp < 2) {
                        if (p1 > phiMaxGeV) phiMaxGeV = p1;
                    } else {
                        // host1 pulled below only when needed
                    }
                }
            }
        }
        if (nComp >= 2) {
            auto host1 = model.fldS(1_c).getLocalNDHostView();
            phiMaxGeV = 0.0;
            for (size_t i = 0; i < static_cast<size_t>(localSizes[0]); ++i) {
                for (size_t j = 0; j < static_cast<size_t>(localSizes[1]); ++j) {
                    for (size_t k = 0; k < static_cast<size_t>(localSizes[2]); ++k) {
                        const double p1 = static_cast<double>(host0(i, j, k)) * fStar_;
                        const double p2 = static_cast<double>(host1(i, j, k)) * fStar_;
                        const double rho = std::sqrt(p1 * p1 + p2 * p2);
                        if (rho > phiMaxGeV) phiMaxGeV = rho;
                    }
                }
            }
        }
#ifdef HAVE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &phiMaxGeV, 1, MPI_DOUBLE, MPI_MAX,
                      static_cast<MPI_Comm>(toolbox->mGroup.getBaseComm()));
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

    template <class Model>
    void reduceToRoot(Model& model, std::vector<float>& buf) const {
#ifdef HAVE_MPI
        auto toolbox = model.getToolBox();
        if (toolbox->getNProcesses() <= 1) return;
        MPI_Comm comm = static_cast<MPI_Comm>(toolbox->mGroup.getBaseComm());
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
        const int nComp = model.activeScalars();
        const size_t n3 = static_cast<size_t>(latticeN_) *
                          static_cast<size_t>(latticeN_) *
                          static_cast<size_t>(latticeN_);

        std::vector<float> buf0(n3), pi0(n3);
        std::vector<float> buf1, pi1;
        if (nComp > 1) {
            buf1.resize(n3);
            pi1.resize(n3);
        }

        packField(model, model.fldS(0_c), buf0);
        reduceToRoot(model, buf0);
        packField(model, model.piS(0_c), pi0);
        reduceToRoot(model, pi0);
        if (nComp > 1) {
            packField(model, model.fldS(1_c), buf1);
            reduceToRoot(model, buf1);
            packField(model, model.piS(1_c), pi1);
            reduceToRoot(model, pi1);
        }

        if (!isRoot) return;

        const std::string fname = "snapshot_" + zeroPad(n, 10) + ".raw";
        const std::string fpath = stateDir_ + fname;

        SnapshotHeader hdr{};
        hdr.magic = (nComp > 1) ? SNAPSHOT_MAGIC_PI2 : SNAPSHOT_MAGIC_PI;
        hdr.N = static_cast<uint32_t>(latticeN_);
        hdr.step = n;
        hdr.t = t;
        hdr.T = model.currentT();
        hdr.a = model.aI;
        hdr.H = snapshotHubble(model);
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
            manifest_ << n << ','
                      << std::setprecision(16) << t << ','
                      << hdr.T << ',' << hdr.a << ',' << hdr.H << ','
                      << fStar_ << ',' << nComp << ','
                      << model.expansionStageId() << ',' << model.rhoMatter() << ','
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

#endif  // THERMAL_INFLATION_FIELD_SNAPSHOT_V2_HPP
