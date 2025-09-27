import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.widgets as widgets
from matplotlib.gridspec import GridSpec

class QRTeachingTool:
    def __init__(self):
        self.A = None
        self.b = None
        self.Q = None
        self.R = None
        self.gs_history = []
        self.current_step = 0
        self.setup_ui()
    
    def gram_schmidt(self, A):
        """Classical Gram-Schmidt with step-by-step history."""
        m, n = A.shape
        Q = np.zeros_like(A, dtype=float)
        R = np.zeros((n, n), dtype=float)
        history = []
        
        for k in range(n):
            # Current column
            a_k = A[:, k].copy()
            r_k = a_k.copy()  # residual
            projections = []
            
            # Project onto previous orthonormal vectors
            for i in range(k):
                proj_coeff = np.dot(Q[:, i], a_k)
                projection = proj_coeff * Q[:, i]
                projections.append(projection)
                r_k -= projection
                R[i, k] = proj_coeff
            
            # Normalize residual
            norm_r_k = np.linalg.norm(r_k)
            if norm_r_k < 1e-12:
                raise ValueError(f"Column {k} is linearly dependent")
            
            q_k = r_k / norm_r_k
            Q[:, k] = q_k
            R[k, k] = norm_r_k
            
            # Store step history
            history.append({
                'k': k,
                'a_k': a_k.copy(),
                'projections': projections.copy(),
                'residual': r_k.copy(),
                'q_k': q_k.copy(),
                'R_partial': R[:k+1, :k+1].copy()
            })
        
        return Q, R, history
    
    def back_substitution(self, R, y):
        """Solve Rx = y for upper triangular R."""
        n = len(y)
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
        return x
    
    def least_squares(self, A, b):
        """Solve least squares using QR decomposition."""
        try:
            # reuse computed Q,R if available; compute otherwise
            if self.Q is None or self.R is None or (self.A is not A):
                Q, R, _ = self.gram_schmidt(A)
            else:
                Q, R = self.Q, self.R
            y = Q.T @ b
            x = self.back_substitution(R, y)
            return x
        except Exception as e:
            print(f"Error in least squares: {e}")
            return None
    
    def parse_matrix_input(self):
        """Get matrix input from user."""
        print("\n" + "="*50)
        print("MATRIX INPUT")
        print("="*50)
        print("Choose an option:")
        print("1. Enter matrix manually")
        print("2. Use random 4×3 matrix")
        print("3. Use default example")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '2':
            self.A = np.random.randn(4, 3)
            print(f"Generated random matrix A:\n{self.A}")
        elif choice == '3':
            self.A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
            print(f"Using default matrix A:\n{self.A}")
        else:
            print("Enter matrix A (rows separated by semicolons, elements by spaces):")
            print("Example: 1 2; 3 4; 5 6")
            text = input("Matrix A: ")
            self.A = self.parse_matrix(text)
            if self.A is None:
                print("Invalid format, using default matrix")
                self.A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        
        # Get vector b
        print(f"\nEnter vector b ({self.A.shape[0]} elements, separated by spaces or semicolons):")
        b_text = input("Vector b: ").strip()
        if not b_text:
            self.b = np.random.randn(self.A.shape[0])
            print(f"Using random vector b: {self.b}")
        else:
            self.b = self.parse_vector(b_text)
            if self.b is None or len(self.b) != self.A.shape[0]:
                self.b = np.random.randn(self.A.shape[0])
                print(f"Invalid format, using random vector b: {self.b}")
    
    def parse_matrix(self, text):
        """Parse matrix from text like '1 2 3; 4 5 6'."""
        try:
            rows = text.strip().split(';')
            matrix = []
            for row in rows:
                if row.strip():
                    matrix.append([float(x) for x in row.strip().split()])
            return np.array(matrix)
        except:
            return None
    
    def parse_vector(self, text):
        """Parse vector from text."""
        try:
            if ';' in text:
                return np.array([float(x) for x in text.split(';')])
            else:
                return np.array([float(x) for x in text.split()])
        except:
            return None
    
    def setup_ui(self):
        """Setup matplotlib-based UI."""
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Matrix visualization area
        self.ax_A = self.fig.add_subplot(gs[0, 0])
        self.ax_Q = self.fig.add_subplot(gs[0, 1])
        self.ax_R = self.fig.add_subplot(gs[0, 2])
        self.ax_info = self.fig.add_subplot(gs[0, 3])
        
        # Gram-Schmidt step visualization
        self.ax_orig = self.fig.add_subplot(gs[1, 0])
        self.ax_proj = self.fig.add_subplot(gs[1, 1])
        self.ax_resid = self.fig.add_subplot(gs[1, 2])
        self.ax_norm = self.fig.add_subplot(gs[1, 3])
        
        # Controls area
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.axis('off')
        
        # Initialize empty plots
        self.clear_all_plots()
        
        # Add initial slider placeholder (will be replaced after QR)
        slider_ax = plt.axes([0.2, 0.02, 0.5, 0.03])
        self.step_slider = widgets.Slider(slider_ax, 'GS Step', 0, 1, valinit=0, 
                                         valfmt='%d', valstep=1)
        self.step_slider.on_changed(self.update_step_visualization)
        
        plt.suptitle('QR Decomposition Teaching Tool', fontsize=16, fontweight='bold')
    
    def clear_all_plots(self):
        """Clear all subplot areas."""
        for ax in [self.ax_A, self.ax_Q, self.ax_R, self.ax_info,
                  self.ax_orig, self.ax_proj, self.ax_resid, self.ax_norm]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

    # ---------- NEW: readable matrix heatmap helper ----------
    def _show_matrix(self, ax, M, title, vmin, vmax):
        """Heatmap with numeric annotations for human readability."""
        ax.clear()
        im = ax.imshow(M, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
        # Overlay numbers if not too large
        if M.size <= 400:
            thr = 0.6 * max(vmax, 1e-12)
            for (i, j), val in np.ndenumerate(M):
                txt_color = 'white' if abs(val) > thr else 'black'
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Value', rotation=270, labelpad=10)
    # ---------------------------------------------------------

    def compute_qr(self):
        """Compute QR decomposition and setup visualization."""
        if self.A is None:
            print("No matrix A provided")
            return
        if len(self.A.shape) != 2 or self.A.shape[0] < self.A.shape[1]:
            print("Matrix must be m×n with m ≥ n")
            return
        try:
            # Compute QR with history
            self.Q, self.R, self.gs_history = self.gram_schmidt(self.A)
            
            # Recreate slider with correct range [0, n-1]
            try:
                self.step_slider.ax.remove()
            except Exception:
                pass
            slider_ax = plt.axes([0.2, 0.02, 0.5, 0.03])
            self.step_slider = widgets.Slider(
                ax=slider_ax, label='GS Step',
                valmin=0, valmax=self.A.shape[1]-1, valinit=0, valstep=1
            )
            self.step_slider.on_changed(self.update_step_visualization)
            
            # Validation
            reconstruction_error = np.linalg.norm(self.A - self.Q @ self.R, 'fro')
            orthogonality_error = np.linalg.norm(self.Q.T @ self.Q - np.eye(self.A.shape[1]), 'fro')
            
            print("\n" + "="*60)
            print("✅ QR DECOMPOSITION COMPLETE!")
            print("="*60)
            print(f"📊 Reconstruction error ‖A - QR‖_F: {reconstruction_error:.2e}")
            print(f"📊 Orthogonality error ‖Q^TQ - I‖_F: {orthogonality_error:.2e}")
            print(f"📐 Matrix A shape: {self.A.shape}")
            print(f"📐 Matrix Q shape: {self.Q.shape}")
            print(f"📐 Matrix R shape: {self.R.shape}")
            
            # Visualize matrices
            self.visualize_matrices()
            
            # Show first step
            self.current_step = 0
            self.update_step_visualization(0)
        except Exception as e:
            print(f"❌ Error in QR decomposition: {e}")
    
    def solve_least_squares(self):
        """Solve least squares problem."""
        if self.Q is None or self.R is None:
            print("❌ Compute QR decomposition first")
            return
        if self.b is None:
            print("❌ Vector b not provided")
            return
        
        # Solve using QR
        x_qr = self.least_squares(self.A, self.b)
        if x_qr is None:
            return
        
        # Compare with numpy
        x_np, residuals, rank, s = np.linalg.lstsq(self.A, self.b, rcond=None)
        
        # Compute residuals
        residual_qr = np.linalg.norm(self.A @ x_qr - self.b)
        residual_np = np.linalg.norm(self.A @ x_np - self.b)
        
        print("\n" + "="*60)
        print("📈 LEAST SQUARES SOLUTION")
        print("="*60)
        print(f"QR solution:    x = {x_qr}")
        print(f"NumPy solution: x = {x_np}")
        print(f"Difference:     ‖x_qr - x_np‖ = {np.linalg.norm(x_qr - x_np):.2e}")
        print(f"QR residual:    ‖Ax - b‖ = {residual_qr:.6f}")
        print(f"NumPy residual: ‖Ax - b‖ = {residual_np:.6f}")
        
        # Show solution method
        print(f"\n🔧 Solution method:")
        print(f"1. Q^T b = {self.Q.T @ self.b}")
        print(f"2. Solve Rx = Q^T b using back substitution")
        print(f"3. Final solution: x = {x_qr}")
    
    def visualize_matrices(self):
        """Visualize A, Q, R with consistent color limits and overlaid values."""
        # Consistent symmetric limits across A, Q, R so colors are comparable
        vmax = 1e-9
        if self.A is not None: vmax = max(vmax, np.max(np.abs(self.A)))
        if self.Q is not None: vmax = max(vmax, np.max(np.abs(self.Q)))
        if self.R is not None: vmax = max(vmax, np.max(np.abs(self.R)))
        vmin = -vmax

        # Draw matrices with annotations
        self._show_matrix(self.ax_A, self.A, "Matrix A", vmin, vmax)
        self._show_matrix(self.ax_Q, self.Q, "Orthogonal Q", vmin, vmax)
        self._show_matrix(self.ax_R, self.R, "Upper Triangular R", vmin, vmax)

        # Info panel
        self.ax_info.clear()
        self.ax_info.text(0.1, 0.8, f'A: {self.A.shape}', transform=self.ax_info.transAxes, fontsize=12)
        self.ax_info.text(0.1, 0.6, f'Q: {self.Q.shape}', transform=self.ax_info.transAxes, fontsize=12)
        self.ax_info.text(0.1, 0.4, f'R: {self.R.shape}', transform=self.ax_info.transAxes, fontsize=12)
        self.ax_info.text(0.1, 0.2, f'b: {self.b.shape}', transform=self.ax_info.transAxes, fontsize=12)
        self.ax_info.set_title('Dimensions', fontweight='bold')
        self.ax_info.axis('off')

        self.fig.canvas.draw_idle()
    
    def update_step_visualization(self, val):
        """Update visualization for specific Gram-Schmidt step."""
        if not self.gs_history:
            return
        k = int(val) if isinstance(val, (int, float)) else int(val)
        k = max(0, min(k, len(self.gs_history) - 1))
        step = self.gs_history[k]
        
        print(f"\n{'='*60}")
        print(f"GRAM-SCHMIDT STEP {k} (Processing column a_{k})")
        print(f"{'='*60}")
        print(f"📐 Original column a_{k}: {step['a_k']}")
        
        if step['projections']:
            print(f"🔄 Projections onto previous q_i:")
            for i, proj in enumerate(step['projections']):
                coeff = np.dot(self.Q[:, i], step['a_k'])
                print(f"   proj_q{i}(a_{k}) = {proj} (coeff: {coeff:.3f})")
        else:
            print("🔄 No previous vectors to project onto")
        
        print(f"📏 Residual r_{k}: {step['residual']}")
        print(f"📏 ‖r_{k}‖ = {np.linalg.norm(step['residual']):.6f}")
        print(f"✅ Normalized q_{k}: {step['q_k']}")
        print(f"\n📊 R matrix after step {k}:")
        print(step['R_partial'])
        
        self.visualize_step(step, k)
    
    def visualize_step(self, step, k):
        """Create visual representation of current Gram-Schmidt step."""
        # Original vector
        self.ax_orig.clear()
        self.ax_orig.bar(range(len(step['a_k'])), step['a_k'], alpha=0.7, color='blue')
        self.ax_orig.set_title(f'Original a_{k}', fontweight='bold')
        self.ax_orig.set_ylabel('Value')
        self.ax_orig.grid(True, alpha=0.3)
        
        # Projections
        self.ax_proj.clear()
        if step['projections']:
            proj_sum = np.sum(step['projections'], axis=0)
            self.ax_proj.bar(range(len(proj_sum)), proj_sum, alpha=0.7, color='red')
            self.ax_proj.set_title('Sum of Projections', fontweight='bold')
        else:
            self.ax_proj.bar(range(len(step['a_k'])), np.zeros_like(step['a_k']), alpha=0.7, color='red')
            self.ax_proj.set_title('No Projections', fontweight='bold')
        self.ax_proj.set_ylabel('Value')
        self.ax_proj.grid(True, alpha=0.3)
        
        # Residual
        self.ax_resid.clear()
        self.ax_resid.bar(range(len(step['residual'])), step['residual'], alpha=0.7, color='orange')
        self.ax_resid.set_title(f'Residual r_{k}', fontweight='bold')
        self.ax_resid.set_ylabel('Value')
        self.ax_resid.grid(True, alpha=0.3)
        
        # Normalized
        self.ax_norm.clear()
        self.ax_norm.bar(range(len(step['q_k'])), step['q_k'], alpha=0.7, color='green')
        self.ax_norm.set_title(f'Normalized q_{k}', fontweight='bold')
        self.ax_norm.set_ylabel('Value')
        self.ax_norm.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self.ax_norm.grid(True, alpha=0.3)
        
        self.fig.canvas.draw()
    
    def run(self):
        """Run the interactive teaching tool."""
        print("🎓 QR Decomposition Teaching Tool")
        print("="*50)
        self.parse_matrix_input()
        print("\nComputing QR decomposition...")
        self.compute_qr()
        print("\nSolving least squares problem...")
        self.solve_least_squares()
        print(f"\n🎮 INTERACTIVE CONTROLS:")
        print(f"• Use the slider at the bottom to explore Gram-Schmidt steps")
        print(f"• Close the plot window when done")
        print(f"• Re-run the script to try different matrices")
        plt.show()

# Run the teaching tool
if __name__ == "__main__":
    tool = QRTeachingTool()
    tool.run()
