import firebase_admin
from firebase_admin import credentials, firestore
from typing import List, Dict, Optional
from datetime import datetime
import json
import logging
import os
import sys
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

# Fix emoji encoding crash on Windows (cp1252 can't encode emoji)
# Reconfigure stdout/stderr to replace unencodable characters
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

class FirebaseService:
    def __init__(self):
        """Initialize Firebase service with credentials.

        Init order (correct order — explicit credentials first, default fallback):
          1. If FIREBASE_CREDENTIAL_PATH is set AND the file exists → use it
             (this is the local development path)
          2. Otherwise fall back to Application Default Credentials
             (this is the production / GCP path — Cloud Run, GCE, etc. inject
             credentials automatically via the metadata server)

        The previous version had this backwards — it called initialize_app() with
        no args first, which appears to "succeed" lazily even when no credentials
        are available. The credential-file fallback was never reached.
        """
        try:
            if not firebase_admin._apps:
                cred_path = os.getenv("FIREBASE_CREDENTIAL_PATH", "").strip()
                if cred_path and os.path.exists(cred_path):
                    print(f"🔑 Using service account file: {cred_path}")
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred)
                    print("✅ Firebase initialized with service account file")
                else:
                    if cred_path:
                        print(f"⚠️ FIREBASE_CREDENTIAL_PATH set but file not found: {cred_path}")
                    print("🔄 Falling back to Application Default Credentials")
                    firebase_admin.initialize_app()
                    print("✅ Firebase initialized with default credentials")

            self.db = firestore.client()
            
            # Test the connection with a timeout
            try:
                print("🔍 Testing Firestore connection...")
                # Try a simple operation to verify connectivity
                test_doc = self.db.collection('_test').document('connection_test')
                test_doc.get()
                print("✅ Firebase Firestore connection verified")
            except Exception as e:
                error_msg = str(e)
                if "database (default) does not exist" in error_msg:
                    print("⚠️ Firestore database not set up. Please enable Firestore in Firebase Console.")
                    print("   Visit: https://console.firebase.google.com/project/escape-ujuzxr/firestore")
                elif "404" in error_msg:
                    print("⚠️ Firestore database not found. Please check your Firebase project configuration.")
                else:
                    print(f"⚠️ Firebase connection test failed: {e}")
                # Continue anyway, might work for actual operations
                
        except Exception as e:
            print(f"❌ Firebase initialization failed: {e}")
            print("🔄 Attempting to continue without Firebase...")
            self.db = None

    def store_chat_session(self, session_id: str, messages: List[BaseMessage], summary: str = "") -> bool:
        """
        Store chat session in Firestore
        
        Args:
            session_id: Unique session identifier
            messages: List of chat messages
            summary: Optional conversation summary
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.db is None:
            print("⚠️ Firebase not available, skipping storage")
            return False
            
        try:
            # Convert messages to serializable format
            serialized_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    serialized_messages.append({
                        'type': 'human',
                        'content': msg.content,
                        'timestamp': datetime.now().isoformat()
                    })
                elif isinstance(msg, AIMessage):
                    serialized_messages.append({
                        'type': 'ai',
                        'content': msg.content,
                        'timestamp': datetime.now().isoformat()
                    })
                elif isinstance(msg, SystemMessage):
                    serialized_messages.append({
                        'type': 'system',
                        'content': msg.content,
                        'timestamp': datetime.now().isoformat()
                    })

            # Store in Firestore
            doc_ref = self.db.collection('chat_sessions').document(session_id)
            doc_ref.set({
                'session_id': session_id,
                'messages': serialized_messages,
                'summary': summary,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
                'message_count': len(serialized_messages)
            })
            
            print(f"✅ Stored chat session {session_id} with {len(serialized_messages)} messages")
            return True
            
        except Exception as e:
            print(f"❌ Failed to store chat session {session_id}: {e}")
            return False

    def get_chat_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve chat session from Firestore
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dict containing session data or None if not found
        """
        if self.db is None:
            print("⚠️ Firebase not available, cannot retrieve session")
            return None
            
        try:
            doc_ref = self.db.collection('chat_sessions').document(session_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                print(f"✅ Retrieved chat session {session_id}")
                return data
            else:
                print(f"⚠️ Chat session {session_id} not found")
                return None
                
        except Exception as e:
            print(f"❌ Failed to retrieve chat session {session_id}: {e}")
            return None

    def update_chat_session(self, session_id: str, messages: List[BaseMessage], summary: str = "") -> bool:
        """
        Update existing chat session
        
        Args:
            session_id: Unique session identifier
            messages: Updated list of chat messages
            summary: Updated conversation summary
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.db is None:
            print("⚠️ Firebase not available, skipping update")
            return False
            
        try:
            # Convert messages to serializable format
            serialized_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    serialized_messages.append({
                        'type': 'human',
                        'content': msg.content,
                        'timestamp': datetime.now().isoformat()
                    })
                elif isinstance(msg, AIMessage):
                    serialized_messages.append({
                        'type': 'ai',
                        'content': msg.content,
                        'timestamp': datetime.now().isoformat()
                    })
                elif isinstance(msg, SystemMessage):
                    serialized_messages.append({
                        'type': 'system',
                        'content': msg.content,
                        'timestamp': datetime.now().isoformat()
                    })

            # Update in Firestore
            doc_ref = self.db.collection('chat_sessions').document(session_id)
            doc_ref.update({
                'messages': serialized_messages,
                'summary': summary,
                'updated_at': firestore.SERVER_TIMESTAMP,
                'message_count': len(serialized_messages)
            })
            
            print(f"✅ Updated chat session {session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update chat session {session_id}: {e}")
            return False

    def delete_chat_session(self, session_id: str) -> bool:
        """
        Delete chat session from Firestore
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.db is None:
            print("⚠️ Firebase not available, cannot delete session")
            return False
            
        try:
            doc_ref = self.db.collection('chat_sessions').document(session_id)
            doc_ref.delete()
            print(f"✅ Deleted chat session {session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete chat session {session_id}: {e}")
            return False

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """
        Get conversation summary for a session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            str: Summary text or None if not found
        """
        if self.db is None:
            print("⚠️ Firebase not available, cannot get summary")
            return None
            
        try:
            doc_ref = self.db.collection('chat_sessions').document(session_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                return data.get('summary', '')
            return None
            
        except Exception as e:
            print(f"❌ Failed to get summary for session {session_id}: {e}")
            return None

    def update_session_summary(self, session_id: str, summary: str) -> bool:
        """
        Update conversation summary for a session
        
        Args:
            session_id: Unique session identifier
            summary: New summary text
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.db is None:
            print("⚠️ Firebase not available, cannot update summary")
            return False
            
        try:
            doc_ref = self.db.collection('chat_sessions').document(session_id)
            doc_ref.update({
                'summary': summary,
                'updated_at': firestore.SERVER_TIMESTAMP
            })
            print(f"✅ Updated summary for session {session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update summary for session {session_id}: {e}")
            return False

    def list_sessions(self, limit: int = 100) -> List[Dict]:
        """
        List recent chat sessions
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session metadata
        """
        if self.db is None:
            print("⚠️ Firebase not available, cannot list sessions")
            return []
            
        try:
            sessions = []
            docs = self.db.collection('chat_sessions').order_by('updated_at', direction=firestore.Query.DESCENDING).limit(limit).stream()
            
            for doc in docs:
                data = doc.to_dict()
                
                # Convert Firestore datetime objects to ISO strings
                created_at = data.get('created_at')
                updated_at = data.get('updated_at')
                
                if created_at and hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                if updated_at and hasattr(updated_at, 'isoformat'):
                    updated_at = updated_at.isoformat()
                
                sessions.append({
                    'session_id': data.get('session_id'),
                    'message_count': data.get('message_count', 0),
                    'created_at': created_at,
                    'updated_at': updated_at,
                    'has_summary': bool(data.get('summary'))
                })
            
            print(f"✅ Retrieved {len(sessions)} sessions")
            return sessions
            
        except Exception as e:
            print(f"❌ Failed to list sessions: {e}")
            return []

    def link_session_to_user(self, session_id: str, user_id: str) -> bool:
        """Add user_id field to an existing chat session document."""
        if self.db is None:
            print("⚠️ Firebase not available, cannot link session")
            return False

        try:
            doc_ref = self.db.collection('chat_sessions').document(session_id)
            doc_ref.update({
                'user_id': user_id,
                'updated_at': firestore.SERVER_TIMESTAMP
            })
            print(f"🔗 Linked session {session_id} to user {user_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to link session {session_id} to user {user_id}: {e}")
            return False

    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict]:
        """List all chat sessions belonging to a user_id."""
        if self.db is None:
            print("⚠️ Firebase not available, cannot list user sessions")
            return []

        try:
            sessions = []
            docs = (self.db.collection('chat_sessions')
                    .where('user_id', '==', user_id)
                    .order_by('updated_at', direction=firestore.Query.DESCENDING)
                    .limit(limit)
                    .stream())

            for doc in docs:
                data = doc.to_dict()
                created_at = data.get('created_at')
                updated_at = data.get('updated_at')

                if created_at and hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                if updated_at and hasattr(updated_at, 'isoformat'):
                    updated_at = updated_at.isoformat()

                sessions.append({
                    'session_id': data.get('session_id'),
                    'message_count': data.get('message_count', 0),
                    'created_at': created_at,
                    'updated_at': updated_at,
                    'has_summary': bool(data.get('summary'))
                })

            print(f"✅ Retrieved {len(sessions)} sessions for user {user_id}")
            return sessions
        except Exception as e:
            print(f"❌ Failed to list sessions for user {user_id}: {e}")
            return []

# Global Firebase service instance
firebase_service = None

def get_firebase_service() -> FirebaseService:
    """Get or create Firebase service instance"""
    global firebase_service
    if firebase_service is None:
        firebase_service = FirebaseService()
    return firebase_service
