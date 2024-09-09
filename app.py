import streamlit as st
from googleapiclient.discovery import build
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()

# YouTube API setup
YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Google Generative AI setup
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Langchain setup
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

def get_video_comments(video_id):
    comments = []
    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    ).execute()

    for item in results["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

def analyze_comments(comments):
    analysis = conversation.predict(input=f"Analyze these YouTube comments and summarize what users are demanding, appreciating, and criticizing: {comments}")
    return analysis

def main():
    st.title("YouTube Comment Analyzer and Chatbot")

    video_url = st.text_input("Enter YouTube Video URL:")
    if video_url:
        video_id = video_url.split("v=")[1]
        comments = get_video_comments(video_id)
        analysis = analyze_comments(comments)

        st.subheader("Comment Analysis")
        st.write(analysis)

        st.subheader("Chat with Comments")
        user_input = st.text_input("Ask a question about the comments:")
        if user_input:
            response = conversation.predict(input=f"Based on the YouTube comments analysis, {user_input}")
            st.write("Response:", response)

if __name__ == "__main__":
    main()
