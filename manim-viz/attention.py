from manim import *
import numpy as np


class AttentionBankVisualization(Scene):
    def setup_axes(self):
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=12,
            y_length=7,
            tips=False,
        )
        plane = NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=12,
            y_length=7,
            background_line_style={"stroke_opacity": 0.25},
        )
        self.add(plane, axes)
        return axes

    def vec_arrow(self, axes, vec, color=WHITE, label=None, label_buff=0.15):
        arr = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(vec[0], vec[1]),
            buff=0,
            stroke_width=6,
            max_tip_length_to_length_ratio=0.12,
            color=color,
        )
        if label is None:
            return arr, None
        lab = Text(label, font_size=30, weight=BOLD, color=color).next_to(
            arr.get_end(), UR, buff=label_buff
        )
        return arr, lab

    def token_box(self, text, color=WHITE):
        t = Text(text, font_size=34, weight=BOLD, color=color)
        box = RoundedRectangle(
            corner_radius=0.2, height=t.height + 0.45, width=t.width + 0.7
        )
        box.set_stroke(color, 3)
        box.set_fill(color, opacity=0.08)
        g = VGroup(box, t)
        t.move_to(box.get_center())
        return g

    def attention_bubble(self, title, lines, width=4.2):
        header = Text(title, font_size=26, weight=BOLD)
        body = VGroup(*[Text(l, font_size=22) for l in lines]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.18
        )
        content = VGroup(header, body).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        panel = RoundedRectangle(
            corner_radius=0.25, width=width, height=content.height + 0.6
        )
        panel.set_stroke(WHITE, 2)
        panel.set_fill(BLACK, opacity=0.65)
        content.move_to(panel.get_center())

        return VGroup(panel, content)

    def shift_with_attention(
        self,
        axes,
        base_vec,
        updates,  # list of dicts: {"token": str, "delta": np.array, "weight": float, "color": Color}
        anchor_point=LEFT * 4 + DOWN * 2.4,
        title="Attention",
        subtitle="bank'",
        sentence_text=None,
        reset_label="bank",
        show_base=True,
    ):
        # Base arrow + label
        base_arrow, base_lab = self.vec_arrow(
            axes, base_vec, color=YELLOW, label=reset_label
        )
        current_vec = base_vec.copy()

        # Current (contextualized) arrow
        cur_arrow, cur_lab = self.vec_arrow(
            axes, current_vec, color=BLUE, label=subtitle
        )

        # Token chips
        chips = VGroup()
        for u in updates:
            chip = self.token_box(u["token"], color=u["color"])
            chips.add(chip)
        chips.arrange(RIGHT, buff=0.35).move_to(anchor_point)

        # Attention panel
        panel = self.attention_bubble(
            title, ["start with: bank", "add tokens → bank shifts"], width=4.6
        ).to_corner(DR, buff=0.25)

        # Optional sentence line
        sent = None
        if sentence_text:
            sent = Text(sentence_text, font_size=32)
            sent.to_edge(UP).shift(DOWN * 0.35)

        # Draw initial state
        if sent:
            self.play(FadeIn(sent, shift=DOWN * 0.1))
        self.play(FadeIn(panel))
        self.play(GrowArrow(base_arrow), FadeIn(base_lab, shift=UP * 0.1))
        if show_base:
            self.play(GrowArrow(cur_arrow), FadeIn(cur_lab, shift=UP * 0.1))
        else:
            self.add(cur_arrow, cur_lab)

        # Show chips (but dim until used)
        for chip in chips:
            chip.set_opacity(0.25)
        self.play(FadeIn(chips, shift=UP * 0.1))

        # Animate updates one by one
        trail = VGroup()
        for i, u in enumerate(updates):
            # Highlight the chip
            chip = chips[i]
            self.play(chip.animate.set_opacity(1.0).scale(1.05), run_time=0.3)
            self.play(chip.animate.scale(1 / 1.05), run_time=0.2)

            # Update attention panel lines
            new_lines = [
                "start with: bank",
                f"add: {u['token']}  (w={u['weight']:.2f})",
                "→ update bank'",
            ]
            new_panel = self.attention_bubble(title, new_lines, width=4.6).move_to(
                panel.get_center()
            )
            self.play(Transform(panel, new_panel), run_time=0.35)

            # Compute next vector
            delta = u["delta"] * u["weight"]
            next_vec = current_vec + delta

            # Show a dashed "movement" line and dot at current endpoint
            dot = Dot(axes.c2p(current_vec[0], current_vec[1]), radius=0.06, color=BLUE)
            move_line = DashedLine(
                axes.c2p(current_vec[0], current_vec[1]),
                axes.c2p(next_vec[0], next_vec[1]),
                dash_length=0.15,
                color=u["color"],
                stroke_width=3,
            )
            self.play(FadeIn(dot, scale=0.8), Create(move_line), run_time=0.35)

            # Transform arrow to the new endpoint
            new_arrow, new_lab = self.vec_arrow(
                axes, next_vec, color=BLUE, label=subtitle
            )
            self.play(
                Transform(cur_arrow, new_arrow),
                Transform(cur_lab, new_lab),
                run_time=0.7,
            )

            trail.add(dot, move_line)
            current_vec = next_vec

            # Slight settle
            self.play(chip.animate.set_opacity(0.6), run_time=0.25)

        # Final panel
        final_panel = self.attention_bubble(
            title,
            ["bank' = bank + Σ (wᵢ · tokenᵢ)", "context-specific meaning"],
            width=4.6,
        ).move_to(panel.get_center())
        self.play(Transform(panel, final_panel), run_time=0.5)

        return {
            "base_arrow": base_arrow,
            "base_lab": base_lab,
            "cur_arrow": cur_arrow,
            "cur_lab": cur_lab,
            "chips": chips,
            "panel": panel,
            "trail": trail,
            "sentence": sent,
        }

    def clear_group(self, gdict):
        # Fade out everything we created for one example
        items = [v for v in gdict.values() if v is not None]
        self.play(*[FadeOut(v) for v in items], run_time=0.7)

    def construct(self):
        axes = self.setup_axes()

        # Base "bank" embedding vector (arbitrary but stable)
        bank = np.array([2.0, 0.6])

        # Example 1: "He sat on the river bank."
        # We add river first, then sat (as requested).
        # Deltas are chosen to push bank' toward "river bank" sense.
        ex1 = self.shift_with_attention(
            axes=axes,
            base_vec=bank,
            updates=[
                {
                    "token": "river",
                    "delta": np.array([-1.2, 1.3]),
                    "weight": 0.75,
                    "color": TEAL,
                },
                {
                    "token": "sat",
                    "delta": np.array([-0.6, 0.2]),
                    "weight": 0.55,
                    "color": GREEN,
                },
            ],
            anchor_point=LEFT * 3.8 + DOWN * 2.8,
            title="Attention (example 1)",
            sentence_text="He sat on the river bank.",
            reset_label="bank",
        )

        self.wait(0.8)
        self.clear_group(ex1)

        # Example 2: "He deposited cash at the bank."
        # Start from bank again, then add cash, then deposited.
        # Deltas push bank' toward financial institution sense.
        ex2 = self.shift_with_attention(
            axes=axes,
            base_vec=bank,
            updates=[
                {
                    "token": "cash",
                    "delta": np.array([1.4, 0.6]),
                    "weight": 0.70,
                    "color": ORANGE,
                },
                {
                    "token": "deposited",
                    "delta": np.array([0.8, 1.0]),
                    "weight": 0.60,
                    "color": RED,
                },
            ],
            anchor_point=LEFT * 3.8 + DOWN * 2.8,
            title="Attention (example 2)",
            sentence_text="He deposited cash at the bank.",
            reset_label="bank",
        )

        self.wait(1.2)
        # Keep final state on screen
        self.play(ex2["chips"].animate.set_opacity(1.0))
        self.wait(1.5)
