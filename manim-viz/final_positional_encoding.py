from manim import (
    Scene, VGroup, Square, Text, MathTex, SurroundingRectangle, Arrow,
    FadeIn, FadeOut, Create, Axes,
    BLUE, GREEN, YELLOW, ORANGE, PURPLE, TEAL, RED, MAROON, WHITE, GRAY_B,
    UP, DOWN, LEFT, RIGHT
)
import numpy as np

class PositionalEncodingVectorAdd(Scene):
    def construct(self):
        # ----------------------------
        # Config
        # ----------------------------
        word = "love"
        d_model = 8
        box_size = 0.75
        box_buff = 0.12

        freq_colors = [BLUE, GREEN, YELLOW, ORANGE, PURPLE, TEAL, RED, MAROON]

        # ----------------------------
        # Helpers
        # ----------------------------
        def make_vector(n, stroke=WHITE, fill=GRAY_B, opacity=0.15):
            boxes = VGroup(*[
                Square(box_size, stroke_color=stroke, stroke_width=2)
                .set_fill(fill, opacity)
                for _ in range(n)
            ])
            boxes.arrange(RIGHT, buff=box_buff)
            return boxes

        def embed_labels(boxes):
            return VGroup(*[
                MathTex(fr"e_{{{i}}}", font_size=26).move_to(boxes[i])
                for i in range(len(boxes))
            ])

        def create_waveform(box, freq_idx, is_sin):
            """Create a small waveform graph inside a box showing sin/cos pattern"""
            # Create a mini axes inside the box
            wave_size = box_size * 0.5  # Make waveform smaller than box
            axes = Axes(
                x_range=[0, 2*np.pi, np.pi/2],
                y_range=[-1.2, 1.2, 0.5],
                x_length=wave_size,
                y_length=wave_size,
                axis_config={"color": freq_colors[freq_idx % len(freq_colors)], 
                           "stroke_width": 1.5,
                           "include_ticks": False,
                           "include_numbers": False},
                tips=False
            )
            axes.move_to(box.get_center())
            
            # Create the waveform with frequency based on freq_idx
            # Higher freq_idx = higher frequency (more cycles)
            freq = freq_idx + 1  # Linear frequency scaling (1, 2, 3, ...)
            
            def sin_wave_func(t):
                return np.sin(freq * t)
            
            def cos_wave_func(t):
                return np.cos(freq * t)
            
            if is_sin:
                wave_func = sin_wave_func
                label = MathTex(r"\sin", font_size=18, color=freq_colors[freq_idx % len(freq_colors)])
            else:
                wave_func = cos_wave_func
                label = MathTex(r"\cos", font_size=18, color=freq_colors[freq_idx % len(freq_colors)])
            
            # Create parametric function for the wave
            wave = axes.plot(
                wave_func,
                x_range=[0, 2*np.pi],
                color=freq_colors[freq_idx % len(freq_colors)],
                stroke_width=2.5
            )
            
            # Position label at top of box
            label.move_to(box.get_top() + 0.15 * DOWN)
            
            return VGroup(axes, wave, label)

        def pe_labels(boxes):
            inner = VGroup()
            waveforms = VGroup()
            for i in range(len(boxes)):
                freq = i // 2
                c = freq_colors[freq % len(freq_colors)]
                boxes[i].set_stroke(c, 2).set_fill(c, 0.18)
                
                # Create waveform visualization
                is_sin = (i % 2 == 0)
                wave_group = create_waveform(boxes[i], freq, is_sin)
                waveforms.add(wave_group)
                inner.add(wave_group)
            return inner

        def sum_labels(boxes):
            """Create labels showing e_i + waveform for each box"""
            inner = VGroup()
            for i in range(len(boxes)):
                freq = i // 2
                c = freq_colors[freq % len(freq_colors)]
                is_sin = (i % 2 == 0)
                
                # Create a smaller waveform (just the wave, no axes or label)
                wave_size = box_size * 0.35
                axes = Axes(
                    x_range=[0, 2*np.pi, np.pi/2],
                    y_range=[-1.2, 1.2, 0.5],
                    x_length=wave_size,
                    y_length=wave_size,
                    axis_config={"color": c, 
                               "stroke_width": 1,
                               "include_ticks": False,
                               "include_numbers": False},
                    tips=False
                )
                
                # Position waveform in the right half of the box
                axes.move_to(boxes[i].get_center() + 0.15 * RIGHT)
                
                # Create the waveform
                freq_val = freq + 1
                
                def sin_wave_func(t):
                    return np.sin(freq_val * t)
                
                def cos_wave_func(t):
                    return np.cos(freq_val * t)
                
                if is_sin:
                    wave_func = sin_wave_func
                else:
                    wave_func = cos_wave_func
                
                wave = axes.plot(
                    wave_func,
                    x_range=[0, 2*np.pi],
                    color=c,
                    stroke_width=2
                )
                
                # Create "e_i +" label on the left
                emb_label = MathTex(fr"e_{{{i}}}+", font_size=20, color=WHITE)
                emb_label.move_to(boxes[i].get_center() + 0.2 * LEFT)
                
                inner.add(VGroup(emb_label, axes, wave))
            return inner

        # ----------------------------
        # Title
        # ----------------------------
        # title = Text("Word Embedding + Positional Encoding", font_size=34)
        # title.to_edge(UP)
        # self.play(Write(title))

        # ----------------------------
        # Embedding vector (top)
        # ----------------------------
        emb_vec = make_vector(d_model)
        emb_inner = embed_labels(emb_vec)
        emb_label = Text(f'Embedding for "{word}"', font_size=26)

        emb = VGroup(emb_label, VGroup(emb_vec, emb_inner))
        emb_label.next_to(emb_vec, UP, buff=0.3)
        emb.move_to(2.8 * UP)  # Increased to move everything up and reduce top gap

        self.play(FadeIn(emb_label), FadeIn(emb_vec), FadeIn(emb_inner, lag_ratio=0.05))

        # ----------------------------
        # Positional encoding vector (middle)
        # ----------------------------
        pe_vec = make_vector(d_model)
        pe_inner = pe_labels(pe_vec)
        pe_label = Text("Positional Encoding (different frequencies)", font_size=26)

        pe = VGroup(pe_label, VGroup(pe_vec, pe_inner))
        pe_label.next_to(pe_vec, UP, buff=0.3)
        pe.next_to(emb, DOWN, buff=0.9)  # Slightly reduced spacing

        self.play(FadeIn(pe_label), FadeIn(pe_vec), FadeIn(pe_inner, lag_ratio=0.05))

        # ----------------------------
        # Plus sign
        # ----------------------------
        plus = MathTex("+", font_size=64)
        plus.move_to(emb_vec.get_bottom() + 0.5 * DOWN)
        self.play(FadeIn(plus))

        # ----------------------------
        # Output vector (bottom)
        # ----------------------------
        out_vec = make_vector(d_model)
        for i in range(d_model):
            out_vec[i].set_stroke(freq_colors[(i // 2) % len(freq_colors)], 2)

        out_inner = sum_labels(out_vec)
        out_label = Text("Positional-aware embedding", font_size=26)

        out = VGroup(out_label, VGroup(out_vec, out_inner))
        out_label.next_to(out_vec, UP, buff=0.3)
        out.next_to(pe, DOWN, buff=1.4)  # Slightly reduced spacing

        # Arrows - positioned far to the sides to completely avoid text overlap
        # Arrow from embedding vector (far left side)
        a1_start = emb_vec.get_left() + 0.5 * LEFT
        a1_end = out_vec.get_left() + 0.5 * LEFT
        a1 = Arrow(a1_start, a1_end, buff=0.1, stroke_width=3)
        
        # Arrow from positional encoding vector (middle) to tip of out_label
        # Make arrow 0.3 times its original length
        a2_start = pe_vec.get_center()
        a2_end_full = out_label.get_bottom()
        direction = a2_end_full - a2_start
        a2_end = a2_start + 0.8 * direction
        a2 = Arrow(a2_start, a2_end, buff=0.1, stroke_width=3)

        # Animate arrow first, then output vector
        self.play(Create(a2))
        self.play(FadeIn(out_label), FadeIn(out_vec), FadeIn(out_inner, lag_ratio=0.05))
        self.wait(2.5)

        # ----------------------------
        # Highlight one frequency pair
        # ----------------------------
        # pair = 1
        # highlight = SurroundingRectangle(
        #     VGroup(pe_vec[2*pair], pe_vec[2*pair+1]),
        #     color=freq_colors[pair], buff=0.08, stroke_width=4
        # )
        # self.play(Create(highlight))
        # self.wait(0.4)
        # self.play(FadeOut(highlight))
        # self.wait(0.6)
