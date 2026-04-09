import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import App from '../App';
import BaselineRetrievalPage from '../pages/BaselineRetrievalPage';
import PlainChatPage from '../pages/PlainChatPage'; // ✅ 新增

/**
 * Routing:
 *  - /            : main app (includes login flow inside App)
 *  - /baseline    : baseline retrieval demo (no login, no logs)
 *  - /plain-chat  : standalone plain chat (reuses main login username)
 */
const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/baseline" element={<BaselineRetrievalPage />} />
      <Route path="/plain-chat" element={<PlainChatPage />} /> {/* ✅ 新增 */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

export default AppRoutes;
