"""
Self-Attention Mechanism Animation using Manim
Run with: manim -pql hello2.py SelfAttentionAnimation
"""

from manim import *
import numpy as np


class SelfAttentionAnimation(Scene):
    def construct(self):
        # Colors
        QUERY_COLOR = "#81c784"  # Green
        KEY_COLOR = "#f8bbd9"  # Pink
        VALUE_COLOR = "#90caf9"  # Blue
        SOFTMAX_COLOR = "#fff59d"  # Yellow
        OUTPUT_COLOR = "#ce93d8"  # Purple

        # Title
        title = Text("Self-Attention", font_size=36)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        # Explain X
        x_explain = Text(
            "X = [He, sat, on, the, river, bank]", font_size=22, color=GRAY
        )
        x_explain.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(x_explain))
        self.wait(0.5)

        # --- TOP ROW: Q and K pairs producing scores ---
        words = ["He", "sat", "on", "the", "river", "bank"]

        # Create Q boxes (all q_bank since we're computing attention for "bank")
        q_boxes = VGroup()
        for i in range(6):
            q_box = VGroup(
                Rectangle(
                    width=0.8,
                    height=0.4,
                    fill_color=QUERY_COLOR,
                    fill_opacity=0.8,
                    stroke_width=1,
                ),
            )
            q_label = Text("q_bank", font_size=10, color=BLACK)
            q_label.move_to(q_box)
            q_boxes.add(VGroup(q_box, q_label))

        # Create K boxes
        k_boxes = VGroup()
        for word in words:
            k_box = VGroup(
                Rectangle(
                    width=0.4,
                    height=0.6,
                    fill_color=KEY_COLOR,
                    fill_opacity=0.8,
                    stroke_width=1,
                ),
            )
            k_label = Text(f"k_{word}", font_size=9, color=BLACK)
            k_label.move_to(k_box)
            k_boxes.add(VGroup(k_box, k_label))

        # Score labels
        score_labels = VGroup()
        for i in range(6):
            s_label = Text(f"s{i + 1}", font_size=14, color=WHITE)
            score_labels.add(s_label)

        # Arrange Q-K pairs horizontally
        qk_pairs = VGroup()
        for i in range(6):
            pair = VGroup(q_boxes[i], k_boxes[i], score_labels[i])
            q_boxes[i].move_to(ORIGIN)
            k_boxes[i].next_to(q_boxes[i], DOWN, buff=0.1)
            score_labels[i].next_to(q_boxes[i], RIGHT, buff=0.1)
            qk_pairs.add(pair)

        qk_pairs.arrange(RIGHT, buff=0.6)
        qk_pairs.move_to(UP * 1.8)

        self.play(
            LaggedStart(*[FadeIn(q) for q in q_boxes], lag_ratio=0.1),
            LaggedStart(*[FadeIn(k) for k in k_boxes], lag_ratio=0.1),
        )
        self.wait(0.3)
        self.play(LaggedStart(*[FadeIn(s) for s in score_labels], lag_ratio=0.1))
        self.wait(0.5)

        # --- Arrows from scores to softmax ---
        softmax_box = VGroup(
            Rectangle(
                width=4,
                height=0.6,
                fill_color=SOFTMAX_COLOR,
                fill_opacity=0.9,
                stroke_width=1,
            ),
        )
        softmax_label = Text("Softmax", font_size=20, color=BLACK)
        softmax_label.move_to(softmax_box)
        softmax_group = VGroup(softmax_box, softmax_label)
        softmax_group.move_to(UP * 0.3)

        # Arrows from scores to softmax
        arrows_to_softmax = VGroup()
        for i, s in enumerate(score_labels):
            arrow = Arrow(
                s.get_bottom(),
                softmax_box.get_top() + LEFT * (2.5 - i) * 0.6,
                buff=0.1,
                color=GRAY,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows_to_softmax.add(arrow)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_to_softmax], lag_ratio=0.05),
            FadeIn(softmax_group),
        )
        self.wait(0.5)

        # --- Arrows from softmax to weights ---
        weight_labels = VGroup()
        for i in range(6):
            w_label = Text(f"w{i + 1}", font_size=14, color=WHITE)
            weight_labels.add(w_label)
        weight_labels.arrange(RIGHT, buff=0.7)
        weight_labels.move_to(DOWN * 0.8)

        arrows_from_softmax = VGroup()
        for i, w in enumerate(weight_labels):
            arrow = Arrow(
                softmax_box.get_bottom() + LEFT * (2.5 - i) * 0.6,
                w.get_top(),
                buff=0.1,
                color=GRAY,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows_from_softmax.add(arrow)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_from_softmax], lag_ratio=0.05),
            LaggedStart(*[FadeIn(w) for w in weight_labels], lag_ratio=0.05),
        )
        self.wait(0.5)

        # --- Value boxes ---
        v_boxes = VGroup()
        for word in words:
            v_box = VGroup(
                Rectangle(
                    width=0.7,
                    height=0.4,
                    fill_color=VALUE_COLOR,
                    fill_opacity=0.8,
                    stroke_width=1,
                ),
            )
            v_label = Text(f"v_{word}", font_size=9, color=BLACK)
            v_label.move_to(v_box)
            v_boxes.add(VGroup(v_box, v_label))

        # Position value boxes below weights
        for i, vb in enumerate(v_boxes):
            vb.next_to(weight_labels[i], DOWN, buff=0.3)

        # Arrows from weights to values
        arrows_to_values = VGroup()
        for i in range(6):
            arrow = Arrow(
                weight_labels[i].get_bottom(),
                v_boxes[i].get_top(),
                buff=0.1,
                color=GRAY,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.2,
            )
            arrows_to_values.add(arrow)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_to_values], lag_ratio=0.05),
            LaggedStart(*[FadeIn(v) for v in v_boxes], lag_ratio=0.05),
        )
        self.wait(0.5)

        # --- Output y_bank ---
        output_box = VGroup(
            Rectangle(
                width=1.2,
                height=0.6,
                fill_color=OUTPUT_COLOR,
                fill_opacity=0.8,
                stroke_width=1,
            ),
        )
        output_label = Text("y_bank", font_size=14, color=BLACK)
        output_label.move_to(output_box)
        output_group = VGroup(output_box, output_label)
        output_group.move_to(DOWN * 2.5)

        arrow_to_output = Arrow(
            v_boxes.get_bottom(),
            output_group.get_top(),
            buff=0.15,
            color=GRAY,
            stroke_width=2,
        )

        self.play(GrowArrow(arrow_to_output), FadeIn(output_group))
        self.wait(0.5)

        # --- Final formula ---
        final_formula = MathTex(
            r"y_{bank} = \sum_i w_i \cdot v_i",
            font_size=26,
            color=WHITE,
        )
        final_box = SurroundingRectangle(
            final_formula, color=YELLOW, buff=0.15, corner_radius=0.1
        )
        final_group = VGroup(final_formula, final_box)
        final_group.to_edge(DOWN, buff=0.2)

        self.play(FadeIn(final_formula), Create(final_box))
        self.wait(2)


if __name__ == "__main__":
    print("Run with: manim -pql hello2.py SelfAttentionAnimation")
