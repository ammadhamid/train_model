# React Native Integration Guide for AI Symptom Checker API

This guide will help frontend developers integrate the AI Symptom Checker API with a React Native app. It covers API usage, authentication, error handling, and best practices.

---

## 1. **API Overview**

- **Base URL:** `http://<your-server>:5000`
- **Endpoints:**
  - `POST /api/analyze-symptom` — Analyze a symptom and get follow-up questions
  - `POST /api/chat` — Multi-turn chat for symptom checking
  - `POST /api/batch-analyze` — Batch analyze multiple symptoms
  - `GET /health` — Health check
- **Authentication:**
  - Use an `X-API-KEY` header (recommended for mobile apps)
  - Or use a JWT in the `Authorization: Bearer <token>` header

---

## 2. **Install Dependencies**

```bash
npm install axios @react-native-async-storage/async-storage
# or
yarn add axios @react-native-async-storage/async-storage
```

---

## 3. **API Service Example (Axios)**

Create `src/services/SymptomCheckerAPI.js`:

```javascript
import axios from 'axios';

const API_BASE_URL = 'http://<your-server>:5000'; // Change for production
const API_KEY = '<your-api-key>'; // Store securely, not in code for production

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
    'X-API-KEY': API_KEY,
  },
});

export const analyzeSymptom = async (symptom, sessionId = null) => {
  const data = { symptom, session_id: sessionId };
  const response = await api.post('/api/analyze-symptom', data);
  return response.data;
};

export const chat = async (message, sessionId, conversationHistory = []) => {
  const data = { message, session_id: sessionId, conversation_history: conversationHistory };
  const response = await api.post('/api/chat', data);
  return response.data;
};

export const batchAnalyze = async (symptoms) => {
  const response = await api.post('/api/batch-analyze', symptoms);
  return response.data;
};
```

---

## 4. **React Native Hook Example**

Create `src/hooks/useSymptomChecker.js`:

```javascript
import { useState, useCallback } from 'react';
import { analyzeSymptom, chat } from '../services/SymptomCheckerAPI';

export const useSymptomChecker = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [sessionId] = useState(`session_${Date.now()}`);

  const analyze = useCallback(async (symptom) => {
    setLoading(true);
    setError(null);
    try {
      const result = await analyzeSymptom(symptom, sessionId);
      setConversationHistory((prev) => [...prev, { type: 'user', message: symptom }, { type: 'ai', message: result }]);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  const sendMessage = useCallback(async (message) => {
    setLoading(true);
    setError(null);
    try {
      const result = await chat(message, sessionId, conversationHistory);
      setConversationHistory((prev) => [...prev, { type: 'user', message }, { type: 'ai', message: result }]);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [sessionId, conversationHistory]);

  const clearConversation = useCallback(() => {
    setConversationHistory([]);
    setError(null);
  }, []);

  return {
    loading,
    error,
    conversationHistory,
    analyze,
    sendMessage,
    clearConversation,
  };
};
```

---

## 5. **Component Example**

```javascript
import React, { useState } from 'react';
import { View, Text, TextInput, Button, ActivityIndicator, ScrollView } from 'react-native';
import { useSymptomChecker } from '../hooks/useSymptomChecker';

const SymptomCheckerScreen = () => {
  const [input, setInput] = useState('');
  const { loading, error, conversationHistory, analyze, sendMessage } = useSymptomChecker();

  const handleAnalyze = async () => {
    if (!input.trim()) return;
    await analyze(input);
    setInput('');
  };

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <ScrollView style={{ flex: 1 }}>
        {conversationHistory.map((msg, idx) => (
          <Text key={idx} style={{ marginVertical: 4, color: msg.type === 'user' ? 'blue' : 'green' }}>
            {msg.type === 'user' ? 'You: ' : 'AI: '}{typeof msg.message === 'string' ? msg.message : JSON.stringify(msg.message)}
          </Text>
        ))}
        {loading && <ActivityIndicator size="large" color="#007AFF" />}
        {error && <Text style={{ color: 'red' }}>{error}</Text>}
      </ScrollView>
      <TextInput
        value={input}
        onChangeText={setInput}
        placeholder="Describe your symptom..."
        style={{ borderWidth: 1, borderColor: '#ccc', borderRadius: 8, padding: 8, marginBottom: 8 }}
      />
      <Button title="Analyze" onPress={handleAnalyze} disabled={loading || !input.trim()} />
    </View>
  );
};

export default SymptomCheckerScreen;
```

---

## 6. **Authentication & Security**
- **API Key:** Store securely (not hardcoded in production). Use environment variables or secure storage.
- **JWT:** If using JWT, obtain from your backend and set in the `Authorization` header.
- **HTTPS:** Always use HTTPS in production.

---

## 7. **Error Handling**
- Catch and display errors from API calls.
- Show user-friendly messages for network or validation errors.
- Handle rate limiting (HTTP 429) gracefully (e.g., show "Too many requests, please wait").

---

## 8. **Best Practices**
- **Debounce** user input if calling API on every keystroke.
- **Validate** input before sending to API.
- **Paginate** or limit conversation history if needed.
- **Test** with both valid and invalid symptoms.
- **Monitor** API health with `/health` endpoint.
- **Update** API base URL and keys for production.

---

## 9. **Testing the Integration**
- Start your backend API (`uvicorn app:app --host 0.0.0.0 --port 5000`)
- Run your React Native app
- Try sending symptoms like "I have a headache" or "I have chest pain"
- Check for follow-up questions and recommendations

---

## 10. **Further Enhancements**
- Add persistent user sessions (store sessionId in AsyncStorage)
- Add push notifications for follow-up
- Support multi-language (translate symptom text before sending)
- Add analytics for usage tracking

---

**For any backend changes (auth, endpoints, etc.), update your frontend integration accordingly!**

---

**Happy coding!** 🚀 