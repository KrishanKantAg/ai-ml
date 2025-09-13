import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import compute_cost, compute_gradient
from lab_utils_uni import plt_intuition, plt_contour_wgrad, plt_gradients


class GradientDescentTutorial:
    """
    Complete gradient descent implementation with visualization tools.
    This class helps you understand and experiment with gradient descent step by step.
    """

    def __init__(self):
        # Initialize with default settings
        self.cost_history = []  # Track cost over iterations
        self.w_history = []  # Track weight parameter changes
        self.b_history = []  # Track bias parameter changes
        self.converged = False  # Track if algorithm converged

    def gradient_descend_test(self, X, y, w_in, b_in, alpha, itr):
        """
        Run the number of itr times and perform a gradient descend to find the best possible value of w and b

        Args:
            X (ndarray): Training data features
            y (ndarray): Training data targets
            w_init (float): Initial weight parameter
            b_init (float): Initial bias parameter
            alpha (float): Learning rate (how big steps we take)
            itr (int): Number of iterations to run
        """
        w = w_in
        b = b_in
        # shape (m)
        m = X.shape[0]

        # cost fn (J_w_b)
        def cost_fn(w, b):
            cost = 0
            for i in range(m):
                cost += ((w * X[i] + b) - y[i]) ** 2
            return (1 / (2 * m)) * cost

        j = 0
        while j < itr:
            # find the cost function and return the w,b associated to it
            costFn = cost_fn(w, b)
            print(f"For itr: {j} -- Cost fn: {costFn} & w: {w}, b: {b}")
            # ----------------------------------------------------
            # Performing 1st level gradient descend
            # ----------------------------------------------------

            # calculating change in w alpha*(dJ_dw)
            change_w = 0
            for i in range(m):
                change_w += ((w * X[i] + b) - y[i]) * X[i]
            change_w = (alpha / m) * change_w
            # calculating change in b alpha*(dJ_db)
            change_b = 0
            for i in range(m):
                change_b += (w * X[i] + b) - y[i]
            change_b = (alpha / m) * change_b
            # break if the values are already converged, ideally we need a minimum threshold here 0 is unrealistic
            if abs(change_w) < 1e-6 and abs(change_b) < 1e-6:
                break
            # Scalar conversion of w and b change
            change_w = change_w.item() if hasattr(change_w, "item") else change_w
            change_b = change_b.item() if hasattr(change_b, "item") else change_b
            w = w - change_w
            b = b - change_b
            j += 1
        return w, b

    def gradient_descent_simple(self, X, y, w_init, b_init, alpha, num_iters):
        """
        Basic gradient descent implementation - great for understanding the core algorithm.

        Args:
            X (ndarray): Training data features
            y (ndarray): Training data targets
            w_init (float): Initial weight parameter
            b_init (float): Initial bias parameter
            alpha (float): Learning rate (how big steps we take)
            num_iters (int): Number of iterations to run

        Returns:
            w (float): Optimized weight
            b (float): Optimized bias
        """
        # Reset tracking arrays for new run
        self.cost_history = []
        self.w_history = []
        self.b_history = []

        # Start with initial parameters
        w = w_init
        b = b_init

        print(f"Starting gradient descent with:")
        print(f"  Initial w: {w_init}, Initial b: {b_init}")
        print(f"  Learning rate (alpha): {alpha}")
        print(f"  Iterations: {num_iters}")
        print("-" * 50)

        # Main gradient descent loop
        for i in range(num_iters):
            # Step 1: Calculate current cost (how wrong our predictions are) */
            cost = compute_cost(X, y, w, b)

            # Step 2: Calculate gradients (which direction to move parameters) */
            dj_db, dj_dw = compute_gradient(X, y, w, b)

            # Step 3: Update parameters using gradients and learning rate */
            w = w - alpha * dj_dw[0]  # Move weight opposite to gradient
            b = b - alpha * dj_db  # Move bias opposite to gradient

            # Ensure all values are scalars for proper formatting */
            cost = cost.item() if hasattr(cost, "item") else cost
            w = w.item() if hasattr(w, "item") else w
            b = b.item() if hasattr(b, "item") else b

            # Step 4: Store history for analysis */
            self.cost_history.append(cost)
            self.w_history.append(w)
            self.b_history.append(b)

            # Print progress every 100 iterations */
            if i % 100 == 0:
                print(
                    f"Iteration {i:4}: Cost = {cost:8.2f}, w = {w:8.3f}, b = {b:8.3f}"
                )

        final_cost = compute_cost(X, y, w, b)
        final_cost = final_cost.item() if hasattr(final_cost, "item") else final_cost
        print("-" * 50)
        print(f"Final results: Cost = {final_cost:8.2f}, w = {w:8.3f}, b = {b:8.3f}")

        return w, b

    def gradient_descent_with_monitoring(
        self, X, y, w_init, b_init, alpha, num_iters, convergence_threshold=1e-6
    ):
        """
        Enhanced gradient descent with convergence monitoring and early stopping.
        This version stops when the algorithm converges (cost stops decreasing significantly).
        """
        # Reset tracking */
        self.cost_history = []
        self.w_history = []
        self.b_history = []
        self.converged = False

        w = w_init
        b = b_init

        print(f"Running enhanced gradient descent...")
        print(f"Will stop early if cost changes less than {convergence_threshold}")

        for i in range(num_iters):
            # Calculate cost and gradients */
            cost = compute_cost(X, y, w, b)
            dj_db, dj_dw = compute_gradient(X, y, w, b)

            # Update parameters */
            w = w - alpha * dj_dw[0]
            b = b - alpha * dj_db

            # Ensure all values are scalars for proper formatting */
            cost = cost.item() if hasattr(cost, "item") else cost
            w = w.item() if hasattr(w, "item") else w
            b = b.item() if hasattr(b, "item") else b

            # Store history */
            self.cost_history.append(cost)
            self.w_history.append(w)
            self.b_history.append(b)

            # Check for convergence (is the cost still decreasing?) */
            if i > 0:
                cost_change = abs(self.cost_history[i - 1] - cost)
                if cost_change < convergence_threshold:
                    print(
                        f"✅ Converged at iteration {i}! Cost change: {cost_change:.2e}"
                    )
                    self.converged = True
                    break

            if i % 100 == 0:
                print(
                    f"Iteration {i:4}: Cost = {cost:8.2f}, w = {w:8.3f}, b = {b:8.3f}"
                )

        return w, b

    def analyze_learning_rate_effects(self, X, y, w_init=0, b_init=0):
        """
        Experiment with different learning rates to see their effects.
        This helps you understand why choosing the right learning rate is crucial.
        """
        # Test different learning rates */
        learning_rates = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
        results = {}

        print("🔬 Analyzing different learning rates...")
        print("=" * 60)

        for alpha in learning_rates:
            print(f"\nTesting learning rate: {alpha}")

            try:
                # Run gradient descent with this learning rate */
                w_final, b_final = self.gradient_descent_simple(
                    X, y, w_init, b_init, alpha, 1000
                )

                # Store results */
                results[alpha] = {
                    "w_final": w_final,
                    "b_final": b_final,
                    "final_cost": self.cost_history[-1],
                    "cost_history": self.cost_history.copy(),
                    "converged": self.cost_history[-1] < self.cost_history[0],
                }

                # Quick assessment */
                if self.cost_history[-1] < self.cost_history[0]:
                    print(f"✅ Learning rate {alpha}: GOOD - Cost decreased")
                else:
                    print(f"❌ Learning rate {alpha}: BAD - Cost increased (diverged)")

            except OverflowError:
                print(f"💥 Learning rate {alpha}: TOO HIGH - Caused overflow!")
                results[alpha] = {"status": "overflow"}

        return results

    def visualize_gradient_descent_path(self, X, y, w_init, b_init, alpha, num_iters):
        """
        Visualize how gradient descent moves through the parameter space.
        This creates the classic "path down the hill" visualization.
        """
        # Run gradient descent and store the path */
        w_final, b_final = self.gradient_descent_simple(
            X, y, w_init, b_init, alpha, num_iters
        )

        # Create visualization */
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Cost over iterations */
        ax1.plot(self.cost_history, "b-", linewidth=2)
        ax1.set_title("Cost Function Over Iterations")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cost J(w,b)")
        ax1.grid(True)

        # Add annotations for key points */
        ax1.annotate(
            f"Start: {self.cost_history[0]:.1f}",
            xy=(0, self.cost_history[0]),
            xytext=(10, self.cost_history[0] + max(self.cost_history) * 0.1),
            arrowprops=dict(arrowstyle="->"),
        )

        ax1.annotate(
            f"End: {self.cost_history[-1]:.1f}",
            xy=(len(self.cost_history) - 1, self.cost_history[-1]),
            xytext=(
                len(self.cost_history) - 20,
                self.cost_history[-1] + max(self.cost_history) * 0.05,
            ),
            arrowprops=dict(arrowstyle="->"),
        )

        # Right plot: Parameter evolution */
        ax2.plot(self.w_history, self.b_history, "r-", linewidth=2, alpha=0.7)
        ax2.scatter(
            self.w_history[0],
            self.b_history[0],
            color="green",
            s=100,
            label="Start",
            zorder=5,
        )
        ax2.scatter(
            self.w_history[-1],
            self.b_history[-1],
            color="red",
            s=100,
            label="End",
            zorder=5,
        )
        ax2.set_title("Parameter Space Journey")
        ax2.set_xlabel("Weight (w)")
        ax2.set_ylabel("Bias (b)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        return w_final, b_final

    def compare_optimizers(self, X, y, w_init=0, b_init=0):
        """
        Compare gradient descent with different strategies.
        Shows you alternatives and when to use them.
        """
        print("🏃‍♂️ Comparing different optimization strategies...")

        # Strategy 1: Conservative (small learning rate) */
        print("\n1. Conservative approach (slow but steady):")
        w1, b1 = self.gradient_descent_with_monitoring(
            X, y, w_init, b_init, 0.01, 10000
        )
        conservative_iters = len(self.cost_history)
        conservative_cost = self.cost_history[-1]

        # Strategy 2: Aggressive (larger learning rate) */
        print("\n2. Aggressive approach (faster but risky):")
        w2, b2 = self.gradient_descent_with_monitoring(X, y, w_init, b_init, 0.1, 10000)
        aggressive_iters = len(self.cost_history)
        aggressive_cost = self.cost_history[-1]

        # Strategy 3: Adaptive (start high, reduce over time) */
        print("\n3. Adaptive approach (start fast, slow down):")
        w3, b3 = self.gradient_descent_adaptive(X, y, w_init, b_init, 0.1, 10000)
        adaptive_iters = len(self.cost_history)
        adaptive_cost = self.cost_history[-1]

        # Summary comparison */
        print("\n" + "=" * 60)
        print("📊 COMPARISON SUMMARY:")
        print("=" * 60)
        print(
            f"Conservative: {conservative_iters:4} iters, cost: {conservative_cost:.6f}"
        )
        print(f"Aggressive:   {aggressive_iters:4} iters, cost: {aggressive_cost:.6f}")
        print(f"Adaptive:     {adaptive_iters:4} iters, cost: {adaptive_cost:.6f}")

        return {
            "conservative": (w1, b1, conservative_cost, conservative_iters),
            "aggressive": (w2, b2, aggressive_cost, aggressive_iters),
            "adaptive": (w3, b3, adaptive_cost, adaptive_iters),
        }

    def gradient_descent_adaptive(self, X, y, w_init, b_init, alpha_init, num_iters):
        """
        Adaptive learning rate gradient descent.
        Starts with a high learning rate and reduces it over time.
        """
        self.cost_history = []
        self.w_history = []
        self.b_history = []

        w = w_init
        b = b_init
        alpha = alpha_init

        for i in range(num_iters):
            # Calculate current state */
            cost = compute_cost(X, y, w, b)
            dj_db, dj_dw = compute_gradient(X, y, w, b)

            # Adaptive learning rate: reduce over time */
            alpha = alpha_init / (1 + i * 0.001)  # Gradually decrease learning rate

            # Update parameters */
            w = w - alpha * dj_dw[0]
            b = b - alpha * dj_db

            # Ensure all values are scalars for proper formatting */
            cost = cost.item() if hasattr(cost, "item") else cost
            w = w.item() if hasattr(w, "item") else w
            b = b.item() if hasattr(b, "item") else b

            # Store history */
            self.cost_history.append(cost)
            self.w_history.append(w)
            self.b_history.append(b)

            # Early stopping if converged */
            if i > 0 and abs(self.cost_history[i - 1] - cost) < 1e-6:
                print(f"Converged at iteration {i}")
                break

            if i % 1000 == 0:
                print(f"Iteration {i:4}: Cost = {cost:8.2f}, alpha = {alpha:.4f}")

        return w, b


# Example usage and experimentation functions */
def run_basic_example():
    """
    Basic example to get you started with gradient descent.
    Uses simple housing price data.
    """
    print("🏠 BASIC HOUSING PRICE EXAMPLE")
    print("=" * 50)

    # Create simple training data */
    # x = house size (in 1000 sq ft), y = price (in $1000s) */
    X_train = np.array([1.0, 2.0, 3.0, 4.0]).reshape(
        -1, 1
    )  # Reshape for matrix operations
    y_train = np.array([300, 500, 700, 900])  # Prices increase with size

    # Initialize our gradient descent tutorial */
    gd = GradientDescentTutorial()

    # Run basic gradient descent */
    print("Running basic gradient descent...")
    w_final, b_final = gd.gradient_descent_simple(
        X_train,
        y_train,
        0,  # Start with weight = 0
        0,  # Start with bias = 0
        0.01,  # Learning rate
        1500,  # Number of iterations
    )

    print(
        f"🥁 Model parameters after gradient descent tuning are - w:{w_final}, b:{b_final}"
    )

    # Test our trained model */
    print(f"\n🎯 TESTING THE TRAINED MODEL:")
    test_sizes = [1.5, 2.5, 3.5]
    for size in test_sizes:
        predicted_price = w_final * size + b_final
        print(
            f"  House size: {size:.1f}k sq ft → Predicted price: ${predicted_price:.0f}k"
        )


def experiment_with_learning_rates():
    """
    Experiment to understand how learning rate affects convergence.
    This is crucial for understanding gradient descent behavior.
    """
    print("\n🧪 LEARNING RATE EXPERIMENT")
    print("=" * 50)

    # Use the same housing data */
    X_train = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    y_train = np.array([300, 500, 700, 900])

    gd = GradientDescentTutorial()

    # Analyze different learning rates */
    results = gd.analyze_learning_rate_effects(X_train, y_train)

    # Plot results for visual comparison */
    fig, ax = plt.subplots(figsize=(12, 6))

    for alpha, result in results.items():
        if "cost_history" in result:
            ax.plot(result["cost_history"][:100], label=f"α = {alpha}", linewidth=2)

    ax.set_title("Cost Function Evolution with Different Learning Rates")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.legend()
    ax.grid(True)
    plt.show()


def visualize_gradient_descent_journey():
    """
    Create beautiful visualizations of the gradient descent process.
    """
    print("\n📊 VISUALIZING THE GRADIENT DESCENT JOURNEY")
    print("=" * 50)

    # Setup data */
    X_train = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    y_train = np.array([300, 500, 700, 900])

    gd = GradientDescentTutorial()

    # Visualize the path gradient descent takes */
    w_final, b_final = gd.visualize_gradient_descent_path(
        X_train,
        y_train,
        w_init=50,  # Start far from optimal
        b_init=200,  # Start far from optimal
        alpha=0.01,  # Good learning rate
        num_iters=1000,
    )


if __name__ == "__main__":
    # Run all examples when script is executed directly */
    print("🚀 GRADIENT DESCENT TUTORIAL - LET'S EXPERIMENT!")
    print("=" * 60)

    # Run basic example first */
    run_basic_example()

    # # Experiment with learning rates */
    #  experiment_with_learning_rates()

    # # Visualize the journey */
    # visualize_gradient_descent_journey()

    # Advanced comparison */
    print("\n🔬 ADVANCED OPTIMIZER COMPARISON")
    print("=" * 50)
    X_train = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    y_train = np.array([300, 500, 700, 900])

    gd = GradientDescentTutorial()
    comparison = gd.compare_optimizers(X_train, y_train)

    print("\n✨ Tutorial complete! Try modifying the parameters and see what happens.")
    print("Key things to experiment with:")
    print("  - Different learning rates (alpha)")
    print("  - Different starting points (w_init, b_init)")
    print("  - Different training data")
    print("  - Number of iterations")
