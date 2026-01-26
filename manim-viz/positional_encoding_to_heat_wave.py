from manim import *
import numpy as np

class PositionalEncodingWavesToHeatmap(Scene):
    def pe_matrix(self, seq_len: int, d_model: int, base: float = 10000.0):
        """
        Returns PE of shape (seq_len, d_model) using the classic sin/cos formula.
        """
        pos = np.arange(seq_len)[:, None]                      # (seq_len, 1)
        i = np.arange(d_model)[None, :]                        # (1, d_model)

        # frequencies for each pair index: 1 / base^(2i/d_model)
        pair_i = (i // 2)                                      # (1, d_model) maps 0,0,1,1,2,2...
        denom = np.power(base, (2.0 * pair_i) / d_model)       # (1, d_model)

        angles = pos / denom                                   # (seq_len, d_model)

        pe = np.zeros_like(angles, dtype=np.float64)
        pe[:, 0::2] = np.sin(angles[:, 0::2])                  # even dims
        pe[:, 1::2] = np.cos(angles[:, 1::2])                  # odd dims
        return pe

    def value_to_color(self, v: float):
        """
        Map [-1, 1] -> color. Feel free to change palette.
        """
        # Normalize to [0,1]
        t = (v + 1.0) / 2.0
        # Interpolate between BLUE (neg) -> WHITE (0) -> RED (pos)
        if t < 0.5:
            return interpolate_color(BLUE, WHITE, t / 0.5)
        else:
            return interpolate_color(WHITE, RED, (t - 0.5) / 0.5)

    def make_heatmap(self, pe: np.ndarray, cell_size=0.10, stroke=0.0):
        """
        Render PE matrix as a VGroup of squares (rows=pos, cols=dim).
        """
        rows, cols = pe.shape
        cells = VGroup()
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=cell_size, stroke_width=stroke)
                sq.set_fill(self.value_to_color(pe[r, c]), opacity=1.0)
                # Place in grid: top-left origin
                sq.move_to(np.array([c * cell_size, -r * cell_size, 0.0]))
                cells.add(sq)

        # Center the whole heatmap
        cells.move_to(ORIGIN)
        return cells

    def make_wave_panel(self, dims_to_show, seq_len, d_model, base=10000.0):
        """
        Create a panel showing sine/cos waves for selected dimensions.
        Each wave is plotted vs position index.
        """
        ax = Axes(
            x_range=[0, seq_len - 1, 5],
            y_range=[-1.2, 1.2, 1],
            x_length=5.0,
            y_length=3.0,
            tips=False,
        )
        ax_labels = ax.get_axis_labels(Tex("pos", font_size=24), Tex("value", font_size=24))

        pe = self.pe_matrix(seq_len, d_model, base=base)

        waves = VGroup()
        labels = VGroup()

        colors = [BLUE, GREEN, YELLOW, ORANGE, PURPLE]
        
        for idx, dim in enumerate(dims_to_show):
            # Plot discrete positions as a smooth curve
            xs = np.arange(seq_len)
            ys = pe[:, dim]

            graph = ax.plot_line_graph(
                x_values=xs,
                y_values=ys,
                add_vertex_dots=False,
                stroke_width=3,
                line_color=colors[idx % len(colors)],
            )
            waves.add(graph)

        panel = VGroup(ax, ax_labels, waves)
        return panel

    def construct(self):
        # --- Parameters ---
        seq_len = 48
        d_model = 64
        base = 10000.0

        # Show many dims spanning early -> late, alternating sin/cos
        # dim 0=sin_0, dim 1=cos_0, dim 2=sin_1, dim 3=cos_1, etc.
        # Show progression from high freq (early) to low freq (late)
        dims_to_show = [
            0, 1,      # sin_0, cos_0 (highest freq)
            2, 3,      # sin_1, cos_1
            4, 5,      # sin_2, cos_2
            6, 7,      # sin_3, cos_3
            10, 11,    # sin_5, cos_5
            14, 15,    # sin_7, cos_7
            20, 21,    # sin_10, cos_10
            26, 27,    # sin_13, cos_13
            30, 31,    # sin_15, cos_15
            40, 41,    # sin_20, cos_20
            50, 51,    # sin_25, cos_25
            62, 63     # sin_31, cos_31 (lowest freq)
        ]

        # --- Setup: Waves on left, Heatmap on right ---
        wave_panel = self.make_wave_panel(dims_to_show, seq_len, d_model, base=base)
        wave_panel.to_edge(LEFT, buff=0.5).shift(UP * 0.5)

        pe = self.pe_matrix(seq_len, d_model, base=base)
        heatmap = self.make_heatmap(pe, cell_size=0.085, stroke=0.0)
        heatmap.to_edge(RIGHT, buff=0.5).shift(UP * 0.5)

        # Extract components for easier access
        ax = wave_panel[0]
        ax_labels = wave_panel[1]
        waves = wave_panel[2]  # VGroup of wave graphs

        # Label below wave graph that shows current sin/cos pair
        current_pair_label = MathTex(r"\sin_0 \quad \cos_0", font_size=32)
        current_pair_label.next_to(wave_panel, DOWN, buff=0.4)

        # Bottom text that will change
        bottom_text = Text("lower dims → high freq", font_size=28)
        bottom_text.to_edge(DOWN, buff=0.5)

        # Build heatmap column by column
        cols = d_model
        col_groups = []
        for c in range(cols):
            col = VGroup(*[heatmap[r * cols + c] for r in range(seq_len)])
            col_groups.append(col)

        # --- Animation: Build waves and heatmap simultaneously ---
        # First show axes
        self.play(
            Create(ax),
            FadeIn(ax_labels),
            FadeIn(current_pair_label),
            FadeIn(bottom_text)
        )
        self.wait(0.2)

        # Track the last column we have made visible to ensure no gaps
        last_shown_col = -1

        # Helper function to get pair index from dimension
        def get_pair_idx(dim):
            return dim // 2
        
        # Helper function to create pair label text
        def make_pair_label(pair_idx):
            return MathTex(fr"\sin_{{{pair_idx}}} \quad \cos_{{{pair_idx}}}", font_size=32)

        def animate_wave_step(wave_idx, run_time=1.0, update_label=True):
            nonlocal last_shown_col, current_pair_label
            dim = dims_to_show[wave_idx]
            pair_idx = get_pair_idx(dim)
            
            # Identify columns to show: everything from last_shown_col+1 up to dim
            cols_to_show = list(range(last_shown_col + 1, dim + 1))
            
            # Prepare animations
            anims = [Create(waves[wave_idx])]
            
            # Fade in heatmap columns
            if cols_to_show:
                grp = VGroup(*[col_groups[c] for c in cols_to_show])
                anims.append(FadeIn(grp, shift=RIGHT * 0.05))
            
            # Update label if needed (usually on even dims / new pairs)
            if update_label and dim % 2 == 0:
                new_label = make_pair_label(pair_idx)
                new_label.move_to(current_pair_label.get_center())
                anims.append(ReplacementTransform(current_pair_label, new_label))
                current_pair_label = new_label
            
            self.play(*anims, run_time=run_time)
            
            last_shown_col = dim

        # Early waves (high frequency): waves 0-7
        for wave_idx in range(8):
            animate_wave_step(wave_idx)
        
        self.wait(0.3)
        
        # Early-mid waves: waves 8-11
        for wave_idx in range(8, 12):
            animate_wave_step(wave_idx)
        
        self.wait(0.3)
        
        # Mid waves: waves 12-15
        for wave_idx in range(12, 16):
            animate_wave_step(wave_idx)
        
        # Note: The original code had a manual block here for cols 28-39.
        # But our new logic inside animate_wave_step will automatically pick up
        # any gap columns when the next wave (wave 16, dim 30) is called.
        # However, looking at dims_to_show: ... 26, 27, 30, 31 ...
        # If we just jump to dim 30, we fill 28, 29, 30.
        # The user's original code filled 28-39 manually BEFORE logic for late waves.
        # But wait, 28-39 includes 30, 31, etc? 
        # Actually dims_to_show has 30, 31.
        # So we should probably let the loop handle it naturally.
        # BUT, the user's original code did a fast fill for a block of columns "mid_cols".
        # If we want to strictly replicate that distinct "fast fill" effect without a wave,
        # we can check if there's a large gap and fill it.
        # BUT, simply relying on the next wave logic is safer for "no gaps".
        # The "mid_cols" in original were 28-39.
        # Wave 16 is dim 30.
        # If we want to show 28, 29 manually before wave 16 starts:
        
        # Let's just proceed with waves. The filling of 28, 29 will happen with wave 16.
        # Any subsequent gaps are also auto-filled.
        
        self.wait(0.3)

        # Prepare text transition for when we show higher dims (low freq)
        new_text = Text("higher dims → low freq", font_size=28)
        new_text.move_to(bottom_text.get_center())
        new_text.set_opacity(0)
        self.add(new_text)

        # Late waves (low frequency): waves 16+
        late_wave_start = 16
        
        for wave_idx in range(late_wave_start, len(waves)):
            dim = dims_to_show[wave_idx]
            
            # Handle text transition on the first late wave
            if wave_idx == late_wave_start:
                 # We need to custom-inject the text animation into the step
                 # Or just do it manually here roughly.
                 # Let's use the helper but add the text anims.
                 
                 # Logic for helper args
                 pair_idx = get_pair_idx(dim)
                 cols_to_show = list(range(last_shown_col + 1, dim + 1))
                 
                 anims = [Create(waves[wave_idx])]
                 if cols_to_show:
                     grp = VGroup(*[col_groups[c] for c in cols_to_show])
                     anims.append(FadeIn(grp, shift=RIGHT * 0.05))
                 
                 if dim % 2 == 0:
                     new_label = make_pair_label(pair_idx)
                     new_label.move_to(current_pair_label.get_center())
                     anims.append(ReplacementTransform(current_pair_label, new_label))
                     current_pair_label = new_label
                 
                 anims.append(bottom_text.animate.set_opacity(0))
                 anims.append(new_text.animate.set_opacity(1))
                 
                 self.play(*anims, run_time=1.2)
                 self.remove(bottom_text)
                 last_shown_col = dim
            else:
                animate_wave_step(wave_idx)
        
        # Fill in any remaining columns up to d_model
        remaining_cols = list(range(last_shown_col + 1, cols))
        chunk = 2
        for i in range(0, len(remaining_cols), chunk):
            cols_chunk = remaining_cols[i:min(i+chunk, len(remaining_cols))]
            grp = VGroup(*[col_groups[c] for c in cols_chunk])
            self.play(FadeIn(grp, shift=RIGHT * 0.05), run_time=0.15)

        self.wait(1.5)
