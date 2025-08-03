import gradio as gr
import pandas as pd
import re
import os
from groq import Groq
import json

# Initialize Groq client
def init_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return Groq(api_key=api_key)
    return None

groq_client = init_groq_client()

# Enhanced toxicity detection using Groq AI
def detect_toxicity_with_groq(text):
    if not groq_client:
        return detect_toxicity_fallback(text)
    
    try:
        prompt = f"""
        You are an AI safety expert specializing in youth digital safety. Analyze the following text for harmful content that could affect young people (ages 13-22).

        Look for:
        - Cyberbullying, harassment, or personal attacks
        - Hate speech or discriminatory language
        - Mental health triggers or harmful content
        - Threats or intimidation
        - Inappropriate sexual content
        - Encouragement of self-harm or dangerous behavior

        Text to analyze: "{text}"

        Respond with a JSON object containing:
        - "is_harmful": true/false
        - "categories": list of detected harmful categories
        - "confidence": 0-100 confidence score
        - "explanation": brief explanation of the analysis
        - "severity": "low", "medium", or "high"

        Be thorough but consider context and intent. False positives should be minimized.
        """

        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"Groq API error: {e}")
        return detect_toxicity_fallback(text)

# Fallback detection for when Groq API is unavailable
def detect_toxicity_fallback(text):
    toxic_keywords = [
        # Cyberbullying terms
        'hate', 'stupid', 'idiot', 'kill', 'die', 'ugly', 'loser', 
        'worthless', 'pathetic', 'disgusting', 'freak', 'bully',
        'nobody likes you', 'kys', 'go kill yourself',
        
        # Hate speech indicators
        'racist', 'sexist', 'homophobic', 'transphobic', 'bigot',
        
        # Mental health triggers
        'depressed', 'suicide', 'self harm', 'cutting', 'pills',
        'end it all', 'not worth living',
        
        # Harassment patterns
        'harassment', 'stalking', 'threatening', 'doxx', 'leaked',
        'embarrassing', 'expose', 'blackmail'
    ]
    
    text_lower = text.lower().strip()
    detected_categories = []
    
    # Check for toxic keywords
    for keyword in toxic_keywords:
        if keyword in text_lower:
            if keyword in ['hate', 'racist', 'sexist', 'homophobic', 'transphobic', 'bigot']:
                detected_categories.append("hate_speech")
            elif keyword in ['depressed', 'suicide', 'self harm', 'cutting', 'pills', 'end it all', 'not worth living']:
                detected_categories.append("mental_health_trigger")
            elif keyword in ['harassment', 'stalking', 'threatening', 'doxx', 'leaked', 'embarrassing', 'expose', 'blackmail']:
                detected_categories.append("harassment")
            else:
                detected_categories.append("cyberbullying")
    
    # Check for patterns like excessive caps (shouting)
    if len(text) > 10 and text.isupper():
        detected_categories.append("aggressive_language")
    
    # Check for repeated characters (like "stupiddddd")
    if re.search(r'(.)\1{3,}', text_lower):
        detected_categories.append("aggressive_language")
    
    is_harmful = len(detected_categories) > 0
    
    return {
        "is_harmful": is_harmful,
        "categories": list(set(detected_categories)),
        "confidence": 85 if is_harmful else 90,
        "explanation": f"Detected patterns: {', '.join(detected_categories)}" if is_harmful else "No harmful content detected",
        "severity": "high" if len(detected_categories) > 2 else "medium" if detected_categories else "low"
    }

# Enhanced RAG Helper with Groq AI for personalized safety advice
class RAGHelper:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def get_groq_advice(self, analysis_result, text):
        if not groq_client:
            return self.retrieve_fallback(analysis_result.get('categories', []))
        
        try:
            categories = ', '.join(analysis_result.get('categories', []))
            severity = analysis_result.get('severity', 'medium')
            
            prompt = f"""
            You are a youth digital safety counselor. A young person has encountered potentially harmful online content. 
            
            Analysis: {analysis_result.get('explanation', '')}
            Detected categories: {categories}
            Severity: {severity}
            
            Provide personalized, empathetic advice that:
            1. Validates their experience
            2. Offers specific, actionable steps
            3. Includes relevant resources
            4. Uses youth-friendly language
            5. Emphasizes they're not alone
            
            Keep advice concise (2-3 paragraphs) and supportive. Include emergency contacts if severity is high.
            """

            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            advice = response.choices[0].message.content
            
            # Get relevant resource link
            link = self.get_resource_link(categories)
            
            return advice, link
            
        except Exception as e:
            print(f"Groq advice error: {e}")
            return self.retrieve_fallback(analysis_result.get('categories', []))

    def get_resource_link(self, categories_str):
        categories = categories_str.lower() if categories_str else ""
        
        if 'mental_health' in categories or 'suicide' in categories:
            return "https://mhanational.org"
        elif 'hate_speech' in categories:
            return "https://help.twitter.com/en/safety-and-security/report-abusive-behavior"
        elif 'harassment' in categories or 'cyberbullying' in categories:
            return "https://www.stopbullying.gov"
        else:
            return "https://www.stopbullying.gov"

    def retrieve_fallback(self, categories):
        query_lower = ' '.join(categories).lower() if categories else 'bullying'
        
        topic_mapping = {
            'cyberbullying': 'Bullying',
            'harassment': 'Bullying',
            'hate_speech': 'Hate Speech',
            'mental_health_trigger': 'Mental Health',
            'aggressive_language': 'Bullying'
        }
        
        best_topic = None
        for category in categories:
            if category in topic_mapping:
                best_topic = topic_mapping[category]
                break
        
        if best_topic:
            matching_row = self.data[self.data["topic"].str.contains(best_topic, case=False, na=False)]
            if not matching_row.empty:
                row = matching_row.iloc[0]
                return row["advice"], row["link"]
        
        # Default to bullying advice
        bullying_row = self.data[self.data["topic"].str.contains("Bullying", case=False, na=False)]
        if not bullying_row.empty:
            row = bullying_row.iloc[0]
            return row["advice"], row["link"]
        
        return "Please reach out to trusted adults or counselors for help with online safety concerns.", "https://www.stopbullying.gov"

# Initialize RAG helper
rag = RAGHelper("cyber_safety.csv")

def analyze_tweet(tweet):
    """Analyze social media content for harmful elements and provide safety guidance."""
    if not tweet.strip():
        return "⚠️ Please enter some content to analyze."
    
    # Use enhanced Groq-powered analysis
    analysis_result = detect_toxicity_with_groq(tweet)
    
    if analysis_result["is_harmful"]:
        # Get personalized advice using Groq AI
        advice, link = rag.get_groq_advice(analysis_result, tweet)
        
        severity_emoji = {
            "high": "🚨",
            "medium": "⚠️", 
            "low": "🔸"
        }
        
        confidence_text = f"Confidence: {analysis_result['confidence']}%"
        categories_text = f"Categories: {', '.join(analysis_result['categories'])}" if analysis_result['categories'] else ""
        
        emergency_section = ""
        if analysis_result['severity'] == 'high':
            emergency_section = """
🆘 Immediate Support:
- Crisis Text Line: Text HOME to 741741
- National Suicide Prevention: 988
- Emergency: 911"""
        
        return f"""{severity_emoji.get(analysis_result['severity'], '⚠️')} Harmful Content Detected

AI Analysis: {analysis_result['explanation']}
{categories_text}
{confidence_text}

💡 Personalized Safety Advice:
{advice}

📚 Resource: {link}{emergency_section}

Remember: Your safety matters. Reach out for help when you need it."""
    else:
        return f"""✅ Content Appears Safe

AI Analysis: {analysis_result['explanation']}
Confidence: {analysis_result['confidence']}%

This content doesn't contain harmful elements. Great job maintaining positive online communication!

🌟 Digital Citizenship Tips:
- Always treat others with kindness and respect online
- Think before you post - words have power
- Report harmful content when you see it
- Support friends who might be experiencing cyberbullying

📱 Stay Safe Online: Continue being a positive digital citizen and help create safer online spaces for everyone."""

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="green",
        neutral_hue="slate"
    ),
    title="SafeTweet - AI-Powered Digital Safety for Youth"
) as iface:
    
    gr.Markdown("""
    # 🛡️ SafeTweet - AI-Powered Digital Safety Platform
    
    **Enhanced with Groq AI for More Accurate Analysis**
    
    SafeTweet uses advanced AI to detect harmful content including cyberbullying, hate speech, misinformation, and mental health triggers. 
    Get personalized safety advice powered by cutting-edge language models.
    
    *Built for Katy Youth Hacks 2025 - Social Good Track*
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="📝 Enter Social Media Content to Analyze",
                placeholder="Paste any social media post, comment, or message here to check for harmful content...",
                lines=4,
                max_lines=10
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Content with AI",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 🎯 What We Detect:
            - 🛡️ Cyberbullying & Harassment
            - ⚠️ Hate Speech & Discrimination  
            - 🔍 Misinformation Patterns
            - 💚 Mental Health Triggers
            - 🚨 Threatening Language
            
            ### 🆘 Need Help?
            - Crisis Text Line: 741741
            - Suicide Prevention: 988
            - Emergency: 911
            """)
    
    output_text = gr.Textbox(
        label="🤖 AI Safety Analysis Results",
        lines=10,
        max_lines=20,
        interactive=False
    )
    
    analyze_btn.click(
        fn=analyze_tweet,
        inputs=input_text,
        outputs=output_text
    )
    
    gr.Markdown("""
    ---
    
    ### 📚 Additional Resources
    
    - [StopBullying.gov](https://www.stopbullying.gov) - Comprehensive anti-bullying resources
    - [Mental Health America](https://mhanational.org) - Mental health support and resources  
    - [FactCheck.org](https://www.factcheck.org) - Fight misinformation with fact-checking
    - [Common Sense Media](https://www.commonsensemedia.org/digital-citizenship) - Digital citizenship guide
    
    **Built with ❤️ for Katy Youth Hacks 2025 | Creating Safer Digital Spaces Together**
    """)

# Launch the app
if __name__ == "__main__":
    iface.launch()