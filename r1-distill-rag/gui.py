# gui.py
import gradio as gr
from typing import Any, List, Optional, Tuple, Dict
import logging
import webbrowser
from threading import Timer
from pathlib import Path
from folder_monitor import FolderMonitor
from thought_logger import ThoughtLogger
from datetime import datetime
import time
import os

class EnhancedRAGUI:
    def __init__(self, processor: Any, doc_processor: Any, vector_manager: Any, logger: logging.Logger):
        self.processor = processor
        self.doc_processor = doc_processor
        self.vector_manager = vector_manager
        self.logger = logger
        self.chain = None
        self.history = []
        self.thought_logger = ThoughtLogger()
        self.folder_monitor = None
        self.current_folder = os.getenv("DATA_DIR", "data")

    def _process_query(self, query: str, history: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """Process a user query and update chat history."""
        try:
            # Start timing and logging
            start_time = datetime.now()
            self.thought_logger.log_query_start(query)
            
            # Create or recreate chain if needed
            try:
                if not self.chain:
                    self.logger.info("Creating new chain...")
                    self.thought_logger.log_step("CHAIN_CREATION", {"status": "creating_new_chain"})
                    self.chain = self.processor.create_chain()
                if not self.chain:
                    raise ValueError("Failed to create chain")
            except Exception as ce:
                self.logger.error(f"Error creating chain: {ce}")
                self.thought_logger.log_error("CHAIN_ERROR", str(ce))
                history.append([query, f"Error: Failed to create processing chain. Please try again."])
                return history, history
            
            self.logger.info(f"Processing query: {query}")
            self.thought_logger.log_step("PROCESSING", {"status": "processing_query"})
            
            response = self.chain.invoke(query)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info("Query processed successfully")
            self.thought_logger.log_query_end(query, True, duration)
            
            # Update history
            history.append([query, response])
            return history, history

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg)
            self.thought_logger.log_error("QUERY_ERROR", str(e))
            self.thought_logger.log_query_end(query, False, duration)
            history.append([query, f"An error occurred: {error_msg}"])
            return history, history

    def _handle_folder_change(self, event_type: str, file_path: Path):
        """Handle document folder changes."""
        try:
            self.logger.info(f"Processing folder change: {event_type} - {file_path}")
            
            # Reload all documents
            documents = self.doc_processor.load_documents(self.current_folder)
            if not documents:
                self.logger.warning("No documents found after folder change")
                return
                
            # Process documents
            chunks = self.doc_processor.process_documents(documents)
            self.logger.info(f"Processed {len(chunks)} chunks from {len(documents)} documents")
            
            # Update vector store
            vectorstore = self.vector_manager.create_or_load_vectorstore(chunks)
            if not vectorstore:
                self.logger.error("Failed to update vector store")
                return
                
            # Update processor's vectorstore
            self.processor.vectorstore = vectorstore
            self.chain = None  # Force chain recreation
            
            self.logger.info("Successfully updated knowledge base")
            
        except Exception as e:
            self.logger.error(f"Error handling folder change: {e}")

    def _select_folder(self, folder_path: str) -> Tuple[str, str]:
        """Handle folder selection."""
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                return "", "❌ Selected folder does not exist"
                
            # Update current folder
            self.current_folder = str(folder_path)
            
            # Start monitoring new folder
            if not self.folder_monitor:
                self.folder_monitor = FolderMonitor(self._handle_folder_change, self.logger)
            self.folder_monitor.start_monitoring(self.current_folder)
            
            # Initial load of documents
            self._handle_folder_change("initial", folder_path)
            
            return self.current_folder, f"✓ Successfully switched to folder: {folder_path}"
            
        except Exception as e:
            self.logger.error(f"Error selecting folder: {e}")
            return "", f"❌ Error selecting folder: {str(e)}"

    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            # Get all document files, including subdirectories
            docs_count = sum(1 for _ in Path(self.current_folder).rglob("*") 
                           if _.is_file() and _.suffix.lower() in 
                           ['.pdf', '.docx', '.txt', '.csv', '.xlsx', '.md'])

            vectorstore_status = "✓ Ready" if (self.processor and self.processor.vectorstore) else "⚠ Not Initialized"
            chain_status = "✓ Ready" if self.chain else "⚠ Not Initialized"
            
            return {
                "Current Data Location": self.current_folder,
                "Document Statistics": {
                    "Total Documents": f"{docs_count} files",
                    "Last Document Update": datetime.now().strftime("%I:%M:%S %p")
                },
                "System Health": {
                    "Knowledge Base": vectorstore_status,
                    "Processing Engine": chain_status
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "System Alert": "⚠ Error getting status",
                "Details": str(e)
            }

    def create_interface(self) -> gr.Blocks:
        """Create an enhanced Gradio interface with chat history and file upload."""
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Smart Document Assistant")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=400,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        query = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about your documents...",
                            lines=3
                        )
                        
                    with gr.Row():
                        submit = gr.Button("Submit", variant="primary")
                        clear = gr.Button("Clear History")
                        
                with gr.Column(scale=1):
                    # Data Source Section
                    gr.Markdown("### 📁 Data Source")
                    folder_input = gr.Textbox(
                        label="Document Location",
                        placeholder="Enter folder path...",
                        value=self.current_folder
                    )
                    with gr.Row():
                        select_folder = gr.Button("📂 Select Folder", variant="secondary")
                        refresh_status = gr.Button("🔄 Refresh", variant="secondary")
                    folder_status = gr.Markdown(
                        value="",
                        visible=True
                    )
                    
                    # System Status Section
                    gr.Markdown("### 📊 System Status")
                    status_md = gr.Markdown("Loading status...")
                        
                    def format_status(status_dict):
                        lines = []
                        
                        # Data Location
                        if "Current Data Location" in status_dict:
                            lines.append("**📁 Data Source**")
                            lines.append(f"Location: {status_dict['Current Data Location']}")
                        
                        # Document Stats
                        if "Document Statistics" in status_dict:
                            lines.append("\n**📑 Document Statistics**")
                            stats = status_dict["Document Statistics"]
                            for key, value in stats.items():
                                lines.append(f"- {key}: {value}")
                        
                        # System Health
                        if "System Health" in status_dict:
                            lines.append("\n**🔧 System Health**")
                            health = status_dict["System Health"]
                            for key, value in health.items():
                                lines.append(f"- {key}: {value}")
                        
                        # Alerts
                        if "System Alert" in status_dict:
                            lines.append("\n**⚠️ System Alert**")
                            lines.append(status_dict["System Alert"])
                            if "Details" in status_dict:
                                lines.append(f"Details: {status_dict['Details']}")
                        
                        return "\n".join(lines)
            
            # Example queries
            with gr.Accordion("📝 Example Questions", open=False):
                gr.Examples(
                    examples=[
                        ["What are the main points discussed in the documents?"],
                        ["Summarize the financial information from all documents."],
                        ["What are the key findings about market trends?"],
                        ["Can you analyze the code structure and dependencies?"]
                    ],
                    inputs=query
                )
            
            # Event handlers
            def update_status():
                current_status = self._get_system_status()
                return format_status(current_status)
            
            # Initial status update
            status_md.value = format_status(self._get_system_status())
            
            # Add refresh button handler
            refresh_status.click(
                fn=update_status,
                outputs=[status_md]
            )
            
            # Add refresh to other events
            select_folder.click(
                fn=self._select_folder,
                inputs=[folder_input],
                outputs=[folder_input, folder_status]
            ).then(
                fn=update_status,
                outputs=[status_md]
            )
            
            submit.click(
                fn=self._process_query,
                inputs=[query, chatbot],
                outputs=[chatbot, chatbot]
            ).then(
                fn=update_status,
                outputs=[status_md]
            )
            
            clear.click(
                fn=lambda: ([], []),
                outputs=[chatbot, chatbot]
            )
            
            # Handle Enter key
            query.submit(
                fn=self._process_query,
                inputs=[query, chatbot],
                outputs=[chatbot, chatbot]
            ).then(
                fn=update_status,
                outputs=[status_md]
            )
            
        return interface

    def launch(self, share: bool = True):
        """Launch the enhanced Gradio interface."""
        interface = self.create_interface()
        
        # Configure server
        server_port = 7860
        server_name = "127.0.0.1"
        
        # Open browser after a short delay
        def open_browser():
            webbrowser.open(f'http://{server_name}:{server_port}')
        
        Timer(1.5, open_browser).start()
        
        # Start folder monitoring
        if not self.folder_monitor:
            self.folder_monitor = FolderMonitor(self._handle_folder_change, self.logger)
            self.folder_monitor.start_monitoring(self.current_folder)
        
        # Launch interface
        try:
            self.logger.info(f"Starting server on {server_name}:{server_port}")
            interface.launch(
                server_name=server_name,
                server_port=server_port,
                share=share,
                prevent_thread_lock=False,
                show_error=True
            )
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
        finally:
            # Cleanup
            if self.folder_monitor:
                self.folder_monitor.stop_monitoring()