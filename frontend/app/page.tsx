"use client";

import React, { useState } from 'react';

export default function Home() {
  const [selectedOption, setSelectedOption] = useState("Steve");
  const [inputText, setInputText] = useState("")
  const [submittedText, setSubmittedText] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    setLoading(true);
  
    try {
      const response = await fetch("http://localhost:8000/inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          { 
            message: inputText, 
            personality: selectedOption.toLowerCase(),
          }
        ),
      });
  
      const data = await response.json();
      setSubmittedText(data.response);
      setInputText("");
    } catch (error) {
      console.error("Error sending message:", error);
      setSubmittedText("Failed to get response.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="flex items-center gap-x-4 m-5">
        <p className="text-gray-600 text-2xl">You're chatting with</p>

        <select
          value={selectedOption}
          onChange={(e) => setSelectedOption(e.target.value)}
          className="p-2 rounded w-64 text-gray-600 border border-amber-400 text-center"
        >
          <option value="Steve">Steve</option>
          <option value="Lola">Lola</option>
          <option value="Michael Scott">Michael Scott</option>
        </select>
      </div>


      <h1 className="font-extrabold text-black text-5xl p-2">Hello</h1>
      <p className="text-gray-600 text-2xl">{selectedOption} writes in this box bellow</p>
      <textarea
          className="w-full max-w-md p-2 mb-4 border rounded bg-gray-200 border-amber-400  text-gray-600"
          rows={4}
          readOnly
          value={submittedText}
      />
      <p className="text-gray-600 text-2xl">And you can write here</p>
      <textarea
          className="w-full max-w-md p-2 border rounded bg-gray-300 border-amber-400 text-gray-600"
          rows={4}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
      />
      <button 
        className="bg-amber-400 text-gray-600 pl-4 pr-4 pt-0.5 pb-0.5 m-2 rounded-xl text-xl"
        onClick={handleSend}
      >
        Send
      </button>
    </div>
  );
}
