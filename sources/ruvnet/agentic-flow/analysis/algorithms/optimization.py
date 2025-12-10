"""
Optimization Algorithms for Maternal Health Analysis

High-performance implementations of algorithms from MATHEMATICAL_MODELS.md
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, gamma, multivariate_normal
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict, Optional
import warnings

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TelomereParams:
    """Parameters for telomere attrition model"""
    α0: float = 25.0      # Baseline attrition (bp/year)
    α1: float = 116.0     # Pregnancy effect (bp/pregnancy)
    α2: float = 50.0      # Stress effect (bp/year per unit)
    α3: float = 15.0      # Age acceleration (bp/year²)
    β: float = 10.0       # Repair capacity (bp/year)

    def validate(self):
        """Ensure all parameters are positive"""
        assert all(getattr(self, p) > 0 for p in ['α0', 'α1', 'α2', 'α3', 'β'])


@dataclass
class ImmuneParams:
    """Parameters for immune senescence model"""
    γ0: float = 0.01      # Baseline decline (/year)
    γ1: float = 0.03      # Pregnancy effect (/pregnancy)
    γ2: float = 0.05      # Inflammation amplification
    μ: float = 0.02       # Recovery rate (/year)
    I_max: float = 1.0    # Maximum capacity

    def validate(self):
        """Ensure all parameters are positive"""
        assert all(getattr(self, p) > 0 for p in ['γ0', 'γ1', 'γ2', 'μ', 'I_max'])


@dataclass
class EpigeneticParams:
    """Parameters for epigenetic aging"""
    κ0: float = 1.0       # Baseline aging (years/year)
    κ1: float = 0.5       # Pregnancy acceleration (years/pregnancy)
    κ2: float = 0.3       # Stress acceleration
    κ3: float = 0.05      # Repair/buffering
    M_max: float = 120.0  # Maximum methylation age

    def validate(self):
        """Ensure all parameters are positive"""
        assert all(getattr(self, p) > 0 for p in ['κ0', 'κ1', 'κ2', 'κ3', 'M_max'])


@dataclass
class MaternalState:
    """Complete maternal physiological state"""
    age: float
    parity: int
    telomere_length: float
    immune_capacity: float
    epigenetic_age: float
    stress_load: float

    def validate(self):
        """Ensure state is physically valid"""
        assert self.age >= 0
        assert self.parity >= 0
        assert self.telomere_length >= 0
        assert 0 <= self.immune_capacity <= 1
        assert 0 <= self.stress_load <= 1


# ============================================================================
# TELOMERE DYNAMICS
# ============================================================================

class TelomereModel:
    """Telomere attrition dynamics"""

    def __init__(self, params: TelomereParams):
        self.params = params
        params.validate()

    def rate(self, T: float, age: float, pregnancy: bool,
             stress: float) -> float:
        """
        Compute telomere attrition rate

        dT/dt = -α0 - α1·P - α2·S - α3·A + β·R(T)
        """
        p = self.params
        P = 1 if pregnancy else 0
        R = self.repair_capacity(T)

        return -p.α0 - p.α1 * P - p.α2 * stress - p.α3 * age + p.β * R

    def repair_capacity(self, T: float, T_max: float = 8000.0) -> float:
        """Repair capacity decreases with telomere shortening"""
        return max(0, T / T_max)

    def simulate(self, T0: float, age_range: Tuple[float, float],
                 pregnancy_times: List[float],
                 stress_func: Callable[[float], float]) -> Dict:
        """
        Simulate telomere trajectory

        Parameters:
        -----------
        T0 : Initial telomere length (bp)
        age_range : (start_age, end_age)
        pregnancy_times : List of ages at pregnancy
        stress_func : Function mapping age to stress [0,1]

        Returns:
        --------
        solution : Dict with 't', 'T', 'age' arrays
        """

        def ode_func(t, y):
            age = t
            T = y[0]

            # Check if currently pregnant (within 0.75 years of pregnancy time)
            pregnancy = any(abs(age - pt) < 0.75 for pt in pregnancy_times)
            stress = stress_func(age)

            dT_dt = self.rate(T, age, pregnancy, stress)
            return [dT_dt]

        # Solve ODE
        sol = solve_ivp(
            ode_func,
            age_range,
            [T0],
            method='LSODA',
            dense_output=True,
            rtol=1e-6,
            atol=1e-8
        )

        if not sol.success:
            warnings.warn(f"ODE solver failed: {sol.message}")

        return {
            't': sol.t,
            'age': sol.t,
            'T': sol.y[0],
            'success': sol.success
        }


# ============================================================================
# IMMUNE SYSTEM DYNAMICS
# ============================================================================

class ImmuneModel:
    """Immune capacity dynamics"""

    def __init__(self, params: ImmuneParams):
        self.params = params
        params.validate()

    def rate(self, I: float, parity: int, inflammation: float) -> float:
        """
        Compute immune capacity rate

        dI/dt = -γ0·I - γ1·P·I - γ2·C·I + μ·(I_max - I)
        """
        p = self.params
        return (-p.γ0 * I - p.γ1 * parity * I - p.γ2 * inflammation * I +
                p.μ * (p.I_max - I))

    def simulate(self, I0: float, age_range: Tuple[float, float],
                 parity_func: Callable[[float], int],
                 inflammation_func: Callable[[float], float]) -> Dict:
        """
        Simulate immune capacity trajectory

        Parameters:
        -----------
        I0 : Initial immune capacity [0,1]
        age_range : (start_age, end_age)
        parity_func : Function mapping age to cumulative parity
        inflammation_func : Function mapping age to inflammation level

        Returns:
        --------
        solution : Dict with 't', 'I' arrays
        """

        def ode_func(t, y):
            I = y[0]
            parity = parity_func(t)
            inflammation = inflammation_func(t)

            dI_dt = self.rate(I, parity, inflammation)
            return [dI_dt]

        sol = solve_ivp(
            ode_func,
            age_range,
            [I0],
            method='LSODA',
            dense_output=True,
            rtol=1e-6,
            atol=1e-8
        )

        return {
            't': sol.t,
            'I': sol.y[0],
            'success': sol.success
        }


# ============================================================================
# STRESS AMPLIFICATION
# ============================================================================

def stress_amplification_factor(parity: int, stress: float,
                                 alpha: float = 0.15,
                                 beta: float = 2.5) -> float:
    """
    Compute stress amplification factor

    SAF(n, s) = 1 + α·n·exp(β·s)

    Parameters:
    -----------
    parity : Number of pregnancies
    stress : Socioeconomic stress index [0,1]
    alpha : Pregnancy multiplier
    beta : Stress exponential factor

    Returns:
    --------
    SAF : Amplification factor (≥ 1)
    """
    return 1 + alpha * parity * np.exp(beta * stress)


# ============================================================================
# INTEGRATED MULTI-SYSTEM SIMULATOR
# ============================================================================

class MaternalHealthSimulator:
    """
    Integrated simulator for all physiological systems
    """

    def __init__(self,
                 telomere_params: TelomereParams,
                 immune_params: ImmuneParams,
                 epigenetic_params: EpigeneticParams):
        self.telomere_params = telomere_params
        self.immune_params = immune_params
        self.epigenetic_params = epigenetic_params

    def dynamics(self, t: float, state: np.ndarray,
                 pregnancy_times: List[float],
                 stress_func: Callable[[float], float]) -> np.ndarray:
        """
        System of ODEs for all biomarkers

        state = [T, I, M, S]
        T: Telomere length
        I: Immune capacity
        M: Methylation age
        S: Stress load
        """
        T, I, M, S = state

        # Check if currently pregnant
        age = t
        pregnancy = any(abs(age - pt) < 0.75 for pt in pregnancy_times)
        parity = sum(1 for pt in pregnancy_times if pt <= age)

        # External stress input
        stress_input = stress_func(age)

        # Telomere dynamics
        dT_dt = (-self.telomere_params.α0 -
                 self.telomere_params.α1 * (1 if pregnancy else 0) -
                 self.telomere_params.α2 * S -
                 self.telomere_params.α3 * age +
                 self.telomere_params.β * max(0, T / 8000))

        # Immune dynamics
        inflammation = 0.1  # Placeholder
        dI_dt = (-self.immune_params.γ0 * I -
                 self.immune_params.γ1 * parity * I -
                 self.immune_params.γ2 * inflammation * I +
                 self.immune_params.μ * (self.immune_params.I_max - I))

        # Epigenetic dynamics
        dM_dt = (self.epigenetic_params.κ0 +
                 self.epigenetic_params.κ1 * (1 if pregnancy else 0) +
                 self.epigenetic_params.κ2 * S -
                 self.epigenetic_params.κ3 *
                 (self.epigenetic_params.M_max - M))

        # Stress dynamics (autoregressive with external input)
        dS_dt = -0.1 * S + stress_input

        return np.array([dT_dt, dI_dt, dM_dt, dS_dt])

    def simulate(self,
                 initial_state: MaternalState,
                 duration: float,
                 pregnancy_times: List[float],
                 stress_func: Callable[[float], float]) -> pd.DataFrame:
        """
        Simulate complete maternal health trajectory

        Parameters:
        -----------
        initial_state : Initial physiological state
        duration : Simulation duration (years)
        pregnancy_times : List of ages at pregnancy
        stress_func : External stress input function

        Returns:
        --------
        trajectory : DataFrame with columns [age, T, I, M, S, health_index]
        """
        initial_state.validate()

        # Initial conditions
        y0 = np.array([
            initial_state.telomere_length,
            initial_state.immune_capacity,
            initial_state.epigenetic_age,
            initial_state.stress_load
        ])

        t_span = (initial_state.age, initial_state.age + duration)

        # Solve ODE system
        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, pregnancy_times, stress_func),
            t_span,
            y0,
            method='LSODA',
            dense_output=True,
            rtol=1e-6,
            atol=1e-8,
            max_step=0.1
        )

        if not sol.success:
            warnings.warn(f"Simulation failed: {sol.message}")

        # Create output DataFrame
        df = pd.DataFrame({
            'age': sol.t,
            'telomere': sol.y[0],
            'immune': sol.y[1],
            'methylation': sol.y[2],
            'stress': sol.y[3]
        })

        # Compute health index
        df['health_index'] = self.compute_health_index(
            df['telomere'].values,
            df['immune'].values,
            df['methylation'].values,
            df['stress'].values
        )

        return df

    def compute_health_index(self, T: np.ndarray, I: np.ndarray,
                             M: np.ndarray, S: np.ndarray,
                             weights: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Compute composite health index

        H = w1·T_norm + w2·I + w3·(1-M_norm) + w4·(1-S_norm)
        """
        if weights is None:
            weights = (0.30, 0.35, 0.20, 0.15)

        w1, w2, w3, w4 = weights

        # Normalize to [0,1]
        T_norm = np.clip(T / 8000, 0, 1)
        I_norm = np.clip(I, 0, 1)
        M_norm = np.clip(M / 120, 0, 1)
        S_norm = np.clip(S, 0, 1)

        return (w1 * T_norm + w2 * I_norm +
                w3 * (1 - M_norm) + w4 * (1 - S_norm))


# ============================================================================
# SURVIVAL AND MORTALITY MODELS
# ============================================================================

class GompertzMakehamModel:
    """Gompertz-Makeham mortality model with maternal factors"""

    def __init__(self, alpha: float = 0.001, beta: float = 0.0001,
                 gamma: float = 0.08, delta: float = 2.0):
        """
        Parameters:
        -----------
        alpha : Background mortality
        beta : Baseline aging rate
        gamma : Aging acceleration
        delta : Health protection factor
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def hazard(self, age: float, parity: int, stress: float,
               health_index: float) -> float:
        """
        Mortality hazard function

        λ(t|n,s,h) = α + β·exp(γ·t)·SAF(n,s)·exp(-δ·h)
        """
        saf = stress_amplification_factor(parity, stress)
        return (self.alpha +
                self.beta * np.exp(self.gamma * age) * saf *
                np.exp(-self.delta * health_index))

    def survival_probability(self, age: float, parity: int,
                            stress: float, health_func: Callable[[float], float],
                            start_age: float = 0) -> float:
        """
        Survival probability from start_age to age

        S(t) = exp(-∫₀ᵗ λ(u) du)
        """
        def integrand(u):
            h = health_func(u)
            return self.hazard(u, parity, stress, h)

        integral, _ = quad(integrand, start_age, age)
        return np.exp(-integral)

    def expected_lifespan(self, parity: int, stress: float,
                         health_func: Callable[[float], float],
                         start_age: float = 0,
                         max_age: float = 120) -> float:
        """
        Expected remaining lifespan

        E[T] = ∫₀^∞ S(t) dt
        """
        def integrand(age):
            return self.survival_probability(
                age, parity, stress, health_func, start_age
            )

        expected, _ = quad(integrand, start_age, max_age)
        return expected


# ============================================================================
# PARAMETER ESTIMATION
# ============================================================================

class ParameterEstimator:
    """Maximum likelihood estimation for model parameters"""

    @staticmethod
    def estimate_telomere_params(data: pd.DataFrame) -> TelomereParams:
        """
        Estimate telomere parameters from longitudinal data

        Data should have columns: age, parity, stress, telomere_obs
        """

        def negative_log_likelihood(params):
            α0, α1, α2, α3, β, σ = params

            if any(p <= 0 for p in params):
                return 1e10  # Invalid parameters

            log_lik = 0

            for _, row in data.iterrows():
                # Simple linear approximation for speed
                # Full ODE would be too slow for optimization
                T_pred = (8000 -  # Initial telomere
                         α0 * row['age'] -
                         α1 * row['parity'] -
                         α2 * row['stress'] * row['age'] -
                         α3 * row['age']**2 / 2 +
                         β * row['age'] * 0.5)  # Average repair

                log_lik += norm.logpdf(row['telomere_obs'],
                                       loc=T_pred, scale=σ)

            return -log_lik

        # Initial guess
        x0 = [25, 116, 50, 15, 10, 100]

        # Optimize
        result = minimize(
            negative_log_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=[(0.1, None)] * 6
        )

        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")

        α0, α1, α2, α3, β, σ = result.x
        return TelomereParams(α0=α0, α1=α1, α2=α2, α3=α3, β=β)

    @staticmethod
    def estimate_immune_params(data: pd.DataFrame) -> ImmuneParams:
        """
        Estimate immune parameters from longitudinal data

        Data should have columns: age, parity, inflammation, immune_obs
        """

        def negative_log_likelihood(params):
            γ0, γ1, γ2, μ, σ = params

            if any(p <= 0 for p in params):
                return 1e10

            log_lik = 0

            for _, row in data.iterrows():
                # Linear approximation
                I_pred = (1.0 -  # Start at maximum
                         γ0 * row['age'] -
                         γ1 * row['parity'] * row['age'] * 0.5 -
                         γ2 * row['inflammation'] * row['age'] * 0.5 +
                         μ * row['age'] * 0.3)

                I_pred = np.clip(I_pred, 0, 1)

                log_lik += norm.logpdf(row['immune_obs'],
                                       loc=I_pred, scale=σ)

            return -log_lik

        x0 = [0.01, 0.03, 0.05, 0.02, 0.05]

        result = minimize(
            negative_log_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=[(0.001, None)] * 5
        )

        γ0, γ1, γ2, μ, σ = result.x
        return ImmuneParams(γ0=γ0, γ1=γ1, γ2=γ2, μ=μ)


# ============================================================================
# RISK STRATIFICATION
# ============================================================================

class RiskStratifier:
    """Compute personalized mortality risk scores"""

    def __init__(self, mortality_model: GompertzMakehamModel):
        self.mortality_model = mortality_model

    def compute_risk_score(self, state: MaternalState,
                          n_bootstrap: int = 1000) -> Tuple[float, Tuple[float, float]]:
        """
        Compute risk score with confidence interval

        Returns:
        --------
        risk_score : Float in [0, 1] (higher = more risk)
        ci : 95% confidence interval (lower, upper)
        """
        # Base risk score
        base_hazard = self.mortality_model.hazard(
            state.age, state.parity, state.stress_load,
            self._compute_health_index(state)
        )

        # Normalize to [0, 1] scale
        # Assume max reasonable hazard is 0.1 (10% annual mortality)
        risk_score = min(base_hazard / 0.1, 1.0)

        # Bootstrap for confidence intervals
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            # Resample with noise
            noisy_state = self._add_noise_to_state(state)
            noisy_hazard = self.mortality_model.hazard(
                noisy_state.age, noisy_state.parity,
                noisy_state.stress_load,
                self._compute_health_index(noisy_state)
            )
            bootstrap_scores.append(min(noisy_hazard / 0.1, 1.0))

        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)

        return risk_score, (ci_lower, ci_upper)

    def _compute_health_index(self, state: MaternalState) -> float:
        """Simple health index computation"""
        T_norm = state.telomere_length / 8000
        I_norm = state.immune_capacity
        M_norm = state.epigenetic_age / 120
        S_norm = state.stress_load

        return (0.30 * T_norm + 0.35 * I_norm +
                0.20 * (1 - M_norm) + 0.15 * (1 - S_norm))

    def _add_noise_to_state(self, state: MaternalState) -> MaternalState:
        """Add Gaussian noise for bootstrap"""
        return MaternalState(
            age=state.age,  # Don't vary age
            parity=state.parity,  # Don't vary parity
            telomere_length=max(0, state.telomere_length +
                               np.random.normal(0, 200)),
            immune_capacity=np.clip(state.immune_capacity +
                                   np.random.normal(0, 0.05), 0, 1),
            epigenetic_age=max(0, state.epigenetic_age +
                              np.random.normal(0, 2)),
            stress_load=np.clip(state.stress_load +
                               np.random.normal(0, 0.05), 0, 1)
        )


# ============================================================================
# MAIN EXECUTION AND TESTING
# ============================================================================

def main():
    """Demonstration of mathematical models"""
    print("=" * 80)
    print("MATERNAL HEALTH MATHEMATICAL MODELS - DEMONSTRATION")
    print("=" * 80)

    # Initialize models
    telomere_params = TelomereParams()
    immune_params = ImmuneParams()
    epigenetic_params = EpigeneticParams()

    print("\n1. Telomere Dynamics")
    print("-" * 40)
    telomere_model = TelomereModel(telomere_params)

    # Simulate telomere trajectory
    pregnancy_times = [25, 27, 30, 33]  # Four pregnancies
    stress_func = lambda age: 0.3 + 0.1 * np.sin(age / 5)  # Varying stress

    tel_result = telomere_model.simulate(
        T0=8000,
        age_range=(20, 70),
        pregnancy_times=pregnancy_times,
        stress_func=stress_func
    )

    print(f"  Initial telomere length: {tel_result['T'][0]:.0f} bp")
    print(f"  Final telomere length: {tel_result['T'][-1]:.0f} bp")
    print(f"  Total attrition: {tel_result['T'][0] - tel_result['T'][-1]:.0f} bp")
    print(f"  Average rate: {(tel_result['T'][0] - tel_result['T'][-1]) / 50:.1f} bp/year")

    print("\n2. Integrated Multi-System Simulation")
    print("-" * 40)

    simulator = MaternalHealthSimulator(
        telomere_params, immune_params, epigenetic_params
    )

    initial_state = MaternalState(
        age=20,
        parity=0,
        telomere_length=8000,
        immune_capacity=0.9,
        epigenetic_age=20,
        stress_load=0.2
    )

    trajectory = simulator.simulate(
        initial_state=initial_state,
        duration=50,
        pregnancy_times=pregnancy_times,
        stress_func=stress_func
    )

    print(f"  Simulated {len(trajectory)} time points")
    print(f"  Final health index: {trajectory['health_index'].iloc[-1]:.3f}")
    print(f"  Health decline: {trajectory['health_index'].iloc[0] - trajectory['health_index'].iloc[-1]:.3f}")

    print("\n3. Survival and Mortality Analysis")
    print("-" * 40)

    mortality_model = GompertzMakehamModel()

    # Compute hazard at different ages
    for age in [30, 50, 70]:
        hazard = mortality_model.hazard(
            age=age,
            parity=len(pregnancy_times),
            stress=0.3,
            health_index=0.7
        )
        print(f"  Mortality hazard at age {age}: {hazard:.6f} (annual risk: {1 - np.exp(-hazard):.4f})")

    print("\n4. Risk Stratification")
    print("-" * 40)

    risk_stratifier = RiskStratifier(mortality_model)

    final_state = MaternalState(
        age=70,
        parity=len(pregnancy_times),
        telomere_length=trajectory['telomere'].iloc[-1],
        immune_capacity=trajectory['immune'].iloc[-1],
        epigenetic_age=trajectory['methylation'].iloc[-1],
        stress_load=trajectory['stress'].iloc[-1]
    )

    risk_score, (ci_lower, ci_upper) = risk_stratifier.compute_risk_score(
        final_state, n_bootstrap=100  # Reduced for speed
    )

    print(f"  Risk score: {risk_score:.3f}")
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    print("\n5. Stress Amplification Factor")
    print("-" * 40)

    for parity in [0, 2, 5, 10]:
        for stress in [0.1, 0.5, 0.9]:
            saf = stress_amplification_factor(parity, stress)
            print(f"  Parity={parity}, Stress={stress:.1f}: SAF={saf:.3f}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
