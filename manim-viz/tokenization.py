"""
Tokenization Visualization - Sentence to Token IDs to Input Embedding
Run with: manim -pqh tokenization.py TokenizationFlow
For video: manim -pqh --format mp4 tokenization.py TokenizationFlow
"""

from manim import (
    Scene,
    Text,
    Table,
    VGroup,
    RoundedRectangle,
    Arrow,
    FadeIn,
    GrowArrow,
    Write,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    BOLD,
    GRAY,
)


class TokenizationFlow(Scene):
    def construct(self):
        # Colors
        TOKEN_COLOR = "#81c784"     # Green
        EMBED_COLOR = "#ffb74d"     # Orange
        ARROW_COLOR = "#90a4ae"     # Gray
        
        # ============ STEP 1: Static sentence at top ============
        input_label = Text("Input Sentence", font_size=20, color=GRAY)
        input_label.to_edge(UP, buff=0.4)
        
        sentence = Text("I love transformers", font_size=36, weight=BOLD)
        sentence.next_to(input_label, DOWN, buff=0.15)
        
        self.add(input_label, sentence)
        self.wait(0.5)
        
        # ============ STEP 2: Vocabulary lookup table ============
        words = ["I", "love", "transformers"]
        token_ids = [42, 891, 2048]
        
        # Create table with strings (Table handles text creation internally)
        table = Table(
            [[word, str(tid)] for word, tid in zip(words, token_ids)],
            col_labels=[Text("Word", font_size=32, weight=BOLD), Text("Token ID", font_size=32, weight=BOLD)],
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": GRAY},
            v_buff=0.3,
            h_buff=0.6,
            element_to_mobject_config={"font_size": 32}
        )
        table.scale(0.75)
        table.move_to(ORIGIN + UP * 0.2)
        
        # Arrow from sentence to table with label
        arrow_to_table = Arrow(
            sentence.get_bottom(),
            table.get_top(),
            buff=0.1,
            color=ARROW_COLOR,
            stroke_width=2
        )
        arrow_label = Text("Token ID Lookup", font_size=18, color=GRAY)
        arrow_label.next_to(arrow_to_table, RIGHT, buff=0.1)
        
        self.play(
            GrowArrow(arrow_to_table),
            Write(arrow_label),
            FadeIn(table),
            run_time=0.8
        )
        self.wait(1)
        
        # ============ STEP 3: Token ID array ============
        token_boxes = VGroup()
        for tid in token_ids:
            text = Text(str(tid), font_size=28, color=TOKEN_COLOR)
            box = RoundedRectangle(
                corner_radius=0.12,
                height=0.7,
                width=0.9,
                stroke_color=TOKEN_COLOR,
                stroke_width=2,
                fill_color=TOKEN_COLOR,
                fill_opacity=0.15
            )
            text.move_to(box.get_center())
            token_boxes.add(VGroup(box, text))
        
        token_boxes.arrange(RIGHT, buff=0.15)
        
        # Array brackets
        left_bracket = Text("[", font_size=40, color=TOKEN_COLOR)
        right_bracket = Text("]", font_size=40, color=TOKEN_COLOR)
        left_bracket.next_to(token_boxes, LEFT, buff=0.1)
        right_bracket.next_to(token_boxes, RIGHT, buff=0.1)
        
        token_array = VGroup(left_bracket, token_boxes, right_bracket)
        token_array.next_to(table, DOWN, buff=0.8)
        
        array_label = Text("Token IDs", font_size=20, color=GRAY)
        array_label.next_to(token_array, LEFT, buff=0.4)
        
        # Arrow from table to token array
        arrow_to_tokens = Arrow(
            table.get_bottom(),
            token_array.get_top(),
            buff=0.1,
            color=ARROW_COLOR,
            stroke_width=2
        )
        
        self.play(
            GrowArrow(arrow_to_tokens),
            FadeIn(token_array),
            FadeIn(array_label),
            run_time=0.8
        )
        self.wait(0.8)
        
        # ============ STEP 4: Input ============
        embed_box = RoundedRectangle(
            corner_radius=0.15,
            height=0.9,
            width=2.0,
            stroke_color=EMBED_COLOR,
            stroke_width=2,
            fill_color=EMBED_COLOR,
            fill_opacity=0.2
        )
        embed_text = Text("Input", font_size=24, color=EMBED_COLOR)
        embed_text.move_to(embed_box.get_center())
        embed_layer = VGroup(embed_box, embed_text)
        embed_layer.next_to(token_array, DOWN, buff=0.5)
        
        arrow = Arrow(
            token_array.get_bottom(),
            embed_layer.get_top(),
            buff=0.08,
            color=ARROW_COLOR,
            stroke_width=2
        )
        
        self.play(
            GrowArrow(arrow),
            FadeIn(embed_layer),
            run_time=0.6
        )
        self.wait(1.5)

