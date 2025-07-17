import json
import os
from datetime import datetime
from typing import Dict

import openai
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

import utils

# Page config
st.set_page_config(
    page_title="Reddit Summary Model Comparison", page_icon="🤖", layout="wide"
)


class ModelManager:
    def __init__(self):
        self.sft_model = None
        self.sft_tokenizer = None
        self.reward_model = None
        self.reward_tokenizer = None
        self.ppo_model = None
        self.ppo_tokenizer = None
        self.device = utils.get_device()

    @st.cache_resource
    def load_models(_self):
        """Load all three models"""
        try:
            # Load SFT model
            _self.sft_tokenizer = AutoTokenizer.from_pretrained(utils.SFT_DIR)
            _self.sft_model = AutoModelForCausalLM.from_pretrained(utils.SFT_DIR).to(
                _self.device
            )

            # Load reward model
            _self.reward_tokenizer = AutoTokenizer.from_pretrained(utils.REWARD_DIR)
            _self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                utils.REWARD_DIR
            ).to(_self.device)

            # Load PPO model
            _self.ppo_tokenizer = AutoTokenizer.from_pretrained(utils.PPO_DIR)
            _self.ppo_model = AutoModelForCausalLM.from_pretrained(utils.PPO_DIR).to(
                _self.device
            )

            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False


class SummaryGenerator:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def format_reddit_post(self, post_text: str) -> str:
        """Format Reddit post text for summarization"""
        # Simple formatting - just return the text as-is since user is pasting it
        return post_text.strip()

    def generate_sft_summary(self, post_text: str) -> str:
        """Generate summary using SFT model"""
        prompt = f"Summarize the following Reddit post:\n\n{post_text}\n\nSummary:"

        inputs = self.model_manager.sft_tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        ).to(self.model_manager.device)

        with torch.no_grad():
            outputs = self.model_manager.sft_model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.model_manager.sft_tokenizer.eos_token_id,
            )

        summary = self.model_manager.sft_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return summary.strip()

    def generate_ppo_summary(self, post_text: str) -> str:
        """Generate summary using PPO model"""
        prompt = f"Summarize the following Reddit post:\n\n{post_text}\n\nSummary:"

        inputs = self.model_manager.ppo_tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        ).to(self.model_manager.device)

        with torch.no_grad():
            outputs = self.model_manager.ppo_model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.model_manager.ppo_tokenizer.eos_token_id,
            )

        summary = self.model_manager.ppo_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return summary.strip()

    def get_reward_score(self, post_text: str, summary: str) -> float:
        """Get reward score for a summary"""
        # Format similar to how the reward model was trained
        input_text = f"Post: {post_text}\nSummary: {summary}"

        inputs = self.model_manager.reward_tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.model_manager.device)

        with torch.no_grad():
            outputs = self.model_manager.reward_model(**inputs)
            score = torch.softmax(outputs.logits, dim=-1)[0][
                1
            ].item()  # Assuming positive class is index 1

        return score

    def compare_summaries(
            self, post_text: str, sft_summary: str, ppo_summary: str
    ) -> Dict:
        """Compare two summaries using reward model"""
        sft_score = self.get_reward_score(post_text, sft_summary)
        ppo_score = self.get_reward_score(post_text, ppo_summary)

        winner = "SFT" if sft_score > ppo_score else "PPO"

        return {
            "sft_score": sft_score,
            "ppo_score": ppo_score,
            "winner": winner,
            "score_difference": abs(sft_score - ppo_score),
        }


class PublicModelAPI:
    def __init__(self):
        self.openai_client = None

    def setup_openai(self):
        """Setup OpenAI client using environment variable"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai.api_key = api_key
        self.openai_client = openai

    def compare_summaries(
            self, post_text: str, sft_summary: str, ppo_summary: str
    ) -> Dict:
        """Have ChatGPT compare the two summaries and pick the better one"""
        try:
            prompt = f"""Given the following Reddit post, please evaluate which of the two summaries is better and explain why.

Reddit Post:
{post_text}

Summary A (SFT Model):
{sft_summary}

Summary B (PPO Model):
{ppo_summary}

Please analyze both summaries and respond with:
1. Which summary is better (A or B)
2. A brief explanation of why it's better

Focus on accuracy, completeness, clarity, and how well each summary captures the key points of the original post."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at evaluating text summaries. Be objective and thorough in your analysis.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=300,
                temperature=0.3,
            )

            response_text = response.choices[0].message.content.strip()

            # Try to parse which summary was chosen
            if (
                    "Summary A" in response_text
                    or "A is better" in response_text
                    or "choose A" in response_text
            ):
                winner = "SFT"
            elif (
                    "Summary B" in response_text
                    or "B is better" in response_text
                    or "choose B" in response_text
            ):
                winner = "PPO"
            else:
                # If we can't parse, look for model names
                if "SFT" in response_text and "PPO" not in response_text:
                    winner = "SFT"
                elif "PPO" in response_text and "SFT" not in response_text:
                    winner = "PPO"
                else:
                    winner = "Unclear"

            return {"winner": winner, "explanation": response_text}

        except Exception as e:
            st.error(f"Error with ChatGPT API: {str(e)}")
            return {"winner": "Error", "explanation": "Error generating comparison"}


def main():
    st.title("🤖 Reddit Summary Model Comparison")
    st.write(
        "Compare SFT, PPO, and Reward Model performance on Reddit post summarization"
    )

    # Initialize components
    model_manager = ModelManager()
    public_api = PublicModelAPI()

    # Check for OpenAI API key
    try:
        public_api.setup_openai()
        st.sidebar.success("✅ OpenAI API key loaded from environment")
    except ValueError as e:
        st.sidebar.error(f"❌ {str(e)}")
        st.error(
            "Please set the OPENAI_API_KEY environment variable to use ChatGPT comparison"
        )

    # Sidebar for configuration
    st.sidebar.title("Configuration")

    # Load models button
    if st.sidebar.button("Load Models"):
        with st.spinner("Loading models..."):
            success = model_manager.load_models()
            if success:
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models")

    # Main interface
    st.header("Reddit Post Analysis")

    # Text input area for Reddit post
    st.subheader("Enter Reddit Post Content")
    reddit_post_text = st.text_area(
        "Paste the Reddit post content here:",
        height=200,
        placeholder="Paste the title, body text, and any relevant comments from a Reddit post here...",
    )

    # Optional: Add some example posts
    st.subheader("Or try an example:")
    example_posts = {
        "Example 1 - Tech Discussion": """Title: Just switched from iPhone to Android after 10 years, here are my thoughts

I've been using iPhones since the iPhone 4 and finally decided to make the switch to Android with the new Samsung Galaxy S24. Here's what I've discovered:

Pros:
- Customization is amazing - I can finally make my phone look exactly how I want
- The camera quality is incredible, especially for night shots
- Battery life is significantly better than my iPhone 13
- File management is so much easier

Cons:
- The ecosystem integration isn't as seamless as Apple's
- Some apps feel less polished
- iMessage withdrawal is real - green bubbles are rough
- Setting everything up took way longer than expected

Overall, I'm happy with the switch but I can see why people stay loyal to either platform. The grass isn't always greener, but the customization options alone make it worth it for me.

Top comments:
- "Welcome to the dark side! Wait until you discover widgets"
- "I made the same switch last year and never looked back"
- "The camera on the S24 is honestly game-changing"
""",
        "Example 2 - Advice Request": """Title: My coworker keeps taking credit for my ideas in meetings, how should I handle this?

I work at a mid-size marketing agency and there's this guy on my team who consistently presents my ideas as his own during client meetings. It's happened at least 5 times in the past month.

Last week I shared a campaign concept with him privately, and then he presented it to the client as his own idea without any attribution. When I tried to speak up, he said "oh yeah, we discussed this together" which made it seem like it was collaborative when it wasn't.

I'm getting really frustrated because it's affecting my visibility with management and clients. I've worked here for 3 years and really like the company otherwise.

Should I:
1. Confront him directly?
2. Talk to my manager?
3. Start documenting everything?
4. Look for a new job?

Any advice would be appreciated.

Top comments:
- "Document everything first, then talk to your manager"
- "I'd start sending follow-up emails after conversations to create a paper trail"
- "This happened to me too - HR was actually really helpful"
""",
    }

    selected_example = st.selectbox(
        "Choose an example:", ["None"] + list(example_posts.keys())
    )

    if selected_example != "None":
        if st.button("Load Example"):
            reddit_post_text = example_posts[selected_example]
            st.rerun()

    if st.button("Analyze Post") and reddit_post_text:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please set the OPENAI_API_KEY environment variable")
            return

        if not all(
                [
                    model_manager.sft_model,
                    model_manager.reward_model,
                    model_manager.ppo_model,
                ]
        ):
            st.error("Please load models first using the sidebar")
            return

        # Initialize summary generator
        summary_generator = SummaryGenerator(model_manager)

        # Display post info
        st.subheader("Post Information")
        st.write(f"**Post Length:** {len(reddit_post_text)} characters")
        st.write(f"**Word Count:** {len(reddit_post_text.split())} words")

        # Show truncated post content
        if len(reddit_post_text) > 300:
            st.write(f"**Content Preview:** {reddit_post_text[:300]}...")
        else:
            st.write(f"**Content:** {reddit_post_text}")

        # Format post for models
        formatted_post = summary_generator.format_reddit_post(reddit_post_text)

        # Generate summaries
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("SFT Model Summary")
            with st.spinner("Generating SFT summary..."):
                sft_summary = summary_generator.generate_sft_summary(formatted_post)
            st.write(sft_summary)

        with col2:
            st.subheader("PPO Model Summary")
            with st.spinner("Generating PPO summary..."):
                ppo_summary = summary_generator.generate_ppo_summary(formatted_post)
            st.write(ppo_summary)

        # Reward model comparison
        st.subheader("Reward Model Evaluation")
        with st.spinner("Comparing summaries..."):
            comparison = summary_generator.compare_summaries(
                formatted_post, sft_summary, ppo_summary
            )

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("SFT Score", f"{comparison['sft_score']:.3f}")
        with col4:
            st.metric("PPO Score", f"{comparison['ppo_score']:.3f}")
        with col5:
            st.metric("Winner", comparison["winner"])

        # ChatGPT comparison
        st.subheader("ChatGPT Evaluation")
        with st.spinner("Getting ChatGPT's comparison..."):
            chatgpt_comparison = public_api.compare_summaries(
                formatted_post, sft_summary, ppo_summary
            )

        st.write(f"**ChatGPT's Choice:** {chatgpt_comparison['winner']}")
        st.write(f"**Explanation:** {chatgpt_comparison['explanation']}")

        # Final comparison table
        st.subheader("Final Results")
        results_data = {
            "Evaluator": ["Reward Model", "ChatGPT"],
            "Winner": [comparison["winner"], chatgpt_comparison["winner"]],
            "Details": [
                f"SFT: {comparison['sft_score']:.3f}, PPO: {comparison['ppo_score']:.3f}",
                "See explanation above",
            ],
        }

        st.dataframe(results_data)

        # Show full summaries in expandable sections
        st.subheader("Full Summaries")

        with st.expander("SFT Full Summary"):
            st.write(sft_summary)

        with st.expander("PPO Full Summary"):
            st.write(ppo_summary)

        # Save results
        if st.button("Save Results"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                "timestamp": timestamp,
                "post_content": reddit_post_text,
                "post_length": len(reddit_post_text),
                "word_count": len(reddit_post_text.split()),
                "sft_summary": sft_summary,
                "ppo_summary": ppo_summary,
                "reward_model_evaluation": {
                    "sft_score": comparison["sft_score"],
                    "ppo_score": comparison["ppo_score"],
                    "winner": comparison["winner"],
                    "score_difference": comparison["score_difference"],
                },
                "chatgpt_evaluation": {
                    "winner": chatgpt_comparison["winner"],
                    "explanation": chatgpt_comparison["explanation"],
                },
            }

            filename = f"results_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)

            st.success(f"Results saved to {filename}")

    # Instructions
    st.sidebar.subheader("Instructions")
    st.sidebar.write(
        """
    1. Set OPENAI_API_KEY environment variable
    2. Load your models using the button above
    3. Paste Reddit post content in the text area
    4. Click 'Analyze Post' to compare summaries
    5. View results and save if desired
    """
    )

    st.sidebar.subheader("Tips")
    st.sidebar.write(
        """
    - Include the post title, body text, and top comments for best results
    - Both reward model and ChatGPT will evaluate the summaries
    - Results are saved as JSON files with timestamps
    - Try the example posts to test the system
    """
    )


if __name__ == "__main__":
    main()
