import React, { useState, useEffect } from 'react';
import useWebSocket from 'react-use-websocket';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface CommandInputProps {
  onSubmit: (command: string) => void;
}

const CommandInput: React.FC<CommandInputProps> = ({ onSubmit }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(input);
    setInput('');
  };

  return (
    <form onSubmit={handleSubmit} className="command-input">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter your trading command..."
      />
      <button type="submit">Send</button>
    </form>
  );
};

const MessageHistory: React.FC<{ messages: Message[] }> = ({ messages }) => {
  return (
    <div className="message-history">
      {messages.map((msg, idx) => (
        <div key={idx} className={`message ${msg.role}`}>
          {msg.content}
        </div>
      ))}
    </div>
  );
};

export const TradingInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const { sendMessage, lastMessage } = useWebSocket('ws://localhost:8000/ws');

  useEffect(() => {
    if (lastMessage) {
      const response = JSON.parse(lastMessage.data);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.message
      }]);
    }
  }, [lastMessage]);

  const handleCommand = (command: string) => {
    setMessages(prev => [...prev, {
      role: 'user',
      content: command
    }]);
    sendMessage(command);
  };

  return (
    <div className="trading-interface">
      <MessageHistory messages={messages} />
      <CommandInput onSubmit={handleCommand} />
    </div>
  );
}; 