import React, { useState, useRef, useEffect } from 'react';
import {
  MessageSquare,
  Send,
  Bot,
  User,
  Sparkles,
  HelpCircle,
  ExternalLink,
  RefreshCw,
  Tag,
  ShieldCheck
} from 'lucide-react';
import axios from 'axios';

export default function AgriChatbot({ currentScanContext, onClearScanContext }) {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      sender: 'bot',
      text: 'Namaste! I am your AgriVision Agricultural Assistant. You can ask me questions about crop disease symptoms, causes, PAU/HAU prevention practices, or general farming guidance in Punjab and Haryana.',
      sources: []
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Context-aware prompt suggestions
  const getSuggestionChips = () => {
    if (currentScanContext && currentScanContext.disease) {
      const { crop, disease } = currentScanContext;
      return [
        `How do I prevent ${crop} ${disease}?`,
        `What causes ${disease} in ${crop}?`,
        `What are PAU guidelines for ${disease}?`,
        `Can this disease spread to neighboring crops?`
      ];
    }
    return [
      'What are the symptoms of yellow rust in wheat?',
      'How to manage bacterial blight in rice?',
      'What fertilizer ratio does PAU recommend for potato?',
      'How to prevent late blight during foggy weather?'
    ];
  };

  const handleSend = async (queryText = null) => {
    const textToSend = (queryText || inputQuery).trim();
    if (!textToSend || isLoading) return;

    const userMessageId = `user-${Date.now()}`;
    const userMessage = {
      id: userMessageId,
      sender: 'user',
      text: textToSend
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputQuery('');
    setIsLoading(true);

    try {
      const payload = {
        question: textToSend,
        context: currentScanContext ? {
          crop: currentScanContext.crop,
          disease: currentScanContext.disease,
          confidence: currentScanContext.confidence
        } : null
      };

      const res = await axios.post('/api/chat', payload);
      const botResponse = {
        id: `bot-${Date.now()}`,
        sender: 'bot',
        text: res.data.answer || 'Thank you for your inquiry. Please consult your local KVK for further details.',
        intent: res.data.intent,
        sources: res.data.sources || []
      };

      setMessages((prev) => [...prev, botResponse]);
    } catch (err) {
      console.error('Chat error:', err);
      setMessages((prev) => [
        ...prev,
        {
          id: `bot-err-${Date.now()}`,
          sender: 'bot',
          text: 'Unable to reach the AgriVision knowledge retrieval service. Please ensure the backend is running.',
          sources: []
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm flex flex-col h-[560px] overflow-hidden">
      {/* Chatbot Header */}
      <div className="px-5 py-4 border-b border-slate-100 bg-slate-50/70 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-emerald-700 text-white rounded-xl shadow-sm">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
              Agricultural Advisory AI
              <span className="text-[10px] bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full font-bold">
                DistilBERT + FAISS
              </span>
            </h3>
            <p className="text-[11px] text-slate-500">
              Grounded in PAU Ludhiana, CCS HAU & ICAR Directives
            </p>
          </div>
        </div>

        {/* Current Active Scan Context Badge */}
        {currentScanContext && currentScanContext.disease && (
          <div className="flex items-center gap-1.5 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-lg text-xs text-emerald-800 font-medium">
            <Tag className="w-3.5 h-3.5 text-emerald-600" />
            <span className="truncate max-w-[140px] sm:max-w-[200px]">
              {currentScanContext.crop}: {currentScanContext.disease}
            </span>
            <button
              onClick={onClearScanContext}
              className="text-emerald-700 hover:text-emerald-900 ml-1 text-[10px] underline"
              title="Clear Scan Context"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 p-4 sm:p-5 overflow-y-auto space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex items-start gap-3 ${
              msg.sender === 'user' ? 'flex-row-reverse' : 'flex-row'
            }`}
          >
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-xs font-bold ${
                msg.sender === 'user'
                  ? 'bg-slate-800 text-white'
                  : 'bg-emerald-600 text-white shadow-sm'
              }`}
            >
              {msg.sender === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            </div>

            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 text-xs sm:text-sm leading-relaxed ${
                msg.sender === 'user'
                  ? 'bg-slate-900 text-white rounded-tr-none'
                  : 'bg-slate-100 text-slate-800 rounded-tl-none border border-slate-200/60'
              }`}
            >
              <p className="whitespace-pre-line">{msg.text}</p>

              {/* Verified Sources Pill */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-2 border-t border-slate-200/60 text-[11px] text-slate-500">
                  <div className="flex items-center gap-1 font-semibold text-slate-600 mb-1">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                    Verified Institutional Sources:
                  </div>
                  <div className="space-y-1">
                    {msg.sources.map((s, idx) => (
                      <div key={idx} className="bg-white/80 px-2 py-1 rounded border border-slate-200 flex items-center justify-between">
                        <span className="font-medium text-slate-700 truncate">{s.institution}</span>
                        {s.url && (
                          <a
                            href={s.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-emerald-700 hover:text-emerald-800 text-[10px] flex items-center gap-0.5 ml-2"
                          >
                            <span>Link</span>
                            <ExternalLink className="w-2.5 h-2.5" />
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-emerald-600 text-white flex items-center justify-center shrink-0">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-slate-100 rounded-2xl rounded-tl-none px-4 py-3 text-xs text-slate-500 flex items-center gap-2 border border-slate-200/60">
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-emerald-600" />
              <span>Searching ICAR & PAU knowledge index...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick Suggestion Chips */}
      <div className="px-4 py-2 border-t border-slate-100 bg-slate-50/50 flex items-center gap-2 overflow-x-auto no-scrollbar">
        <Sparkles className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
        {getSuggestionChips().map((chip, idx) => (
          <button
            key={idx}
            onClick={() => handleSend(chip)}
            disabled={isLoading}
            className="whitespace-nowrap text-[11px] bg-white hover:bg-emerald-50 hover:text-emerald-800 hover:border-emerald-300 text-slate-600 px-3 py-1 rounded-full border border-slate-200 transition shadow-2xs font-medium"
          >
            {chip}
          </button>
        ))}
      </div>

      {/* Input Form Bar */}
      <div className="p-3 bg-white border-t border-slate-100 flex items-center gap-2">
        <input
          type="text"
          value={inputQuery}
          onChange={(e) => setInputQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about crop diseases, symptoms, or farming..."
          disabled={isLoading}
          className="flex-1 bg-slate-100 focus:bg-white text-xs sm:text-sm text-slate-800 placeholder-slate-400 px-4 py-2.5 rounded-xl border border-transparent focus:border-emerald-500 focus:outline-none transition"
        />
        <button
          onClick={() => handleSend()}
          disabled={!inputQuery.trim() || isLoading}
          className="p-2.5 bg-emerald-700 hover:bg-emerald-800 disabled:bg-slate-200 text-white rounded-xl transition shadow-sm active:scale-95 shrink-0"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
