import gradio as gr
import pandas as pd
import random
import json

# Load texts from JSONL file
texts = {}
elo_ratings = {}

# Initialize ID counter
id_counter = 0

# Load texts from JSONL file
with open("samples4.json", "r") as f:
    for line in f:
        id_counter += 1
        sample = json.loads(line)
        text_id = str(id_counter)
        text_content = sample.get("prompts") + "*"*100 + "\n" + sample.get("genPrompt")  # Assuming "prompts" is the key for the text content
        if text_id and text_content:
            texts[text_id] = text_content
            elo_ratings[text_id] = 1200

# Keep track of current pair
current_pair = random.sample(list(texts.keys()), 2)
selection_log = []

def display_texts():
    text_a = texts[current_pair[0]]
    text_b = texts[current_pair[1]]
    return text_a, text_b

def update_scores(winner):
    global current_pair, elo_ratings, selection_log

    if winner == "left":
        winner_text, loser_text = current_pair[0], current_pair[1]
    elif winner == "right":
        winner_text, loser_text = current_pair[1], current_pair[0]

    # Calculate new ELO ratings
    elo_ratings[winner_text] += 10
    elo_ratings[loser_text] -= 10

    # Log the selection
    selection_log.append(f"{current_pair[0]}>{current_pair[1]}" if winner == "left" else f"{current_pair[1]}>{current_pair[0]}")

    # Update current pair to new random texts
    current_pair = random.sample(list(texts.keys()), 2)

    # Prepare scoreboard and log output
    scoreboard = "\n".join([f"{text} - {elo:.0f} ELO" for text, elo in elo_ratings.items()])
    log_output = "\n".join(selection_log)
    return display_texts()[0], display_texts()[1], scoreboard, log_output

def gradio_ui():
    with gr.Blocks() as demo:        
        with gr.Row():
            with gr.Column(scale=4):
                text_a = gr.Textbox(lines=10, label="Text A")
            with gr.Column(scale=4):
                text_b = gr.Textbox(lines=10, label="Text B")

        with gr.Row():
            left_btn = gr.Button("Left (A)")
            right_btn = gr.Button("Right (D)")

        with gr.Row():
            with gr.Column():
                scoreboard = gr.Textbox(lines=10, label="Scoreboard", max_lines=10, elem_id="scoreboard")
            with gr.Column():
                log_output = gr.Textbox(lines=10, label="Selection Log", max_lines=10, elem_id="log_output")

        def update_ui(winner):
            text_a_val, text_b_val, scores, log = update_scores(winner)
            return text_a_val, text_b_val, scores, log

        left_btn.click(fn=update_ui, inputs=[gr.State("left")], outputs=[text_a, text_b, scoreboard, log_output])
        right_btn.click(fn=update_ui, inputs=[gr.State("right")], outputs=[text_a, text_b, scoreboard, log_output])

        text_a.value, text_b.value = display_texts()
        scoreboard.value = "\n".join([f"{text} - {elo:.0f} ELO" for text, elo in elo_ratings.items()])
        log_output.value = "\n".join(selection_log)

    demo.launch()

gradio_ui()
