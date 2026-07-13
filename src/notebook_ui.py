import json
import html
import uuid
import logging

import mlflow
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from src.llm import normalize_content
from src.styles import ENHANCED_CSS
from src.stream import extract_messages
from src.conversation import _tag_trace
from src.agents import (
    reset_inventory, set_item_stock, get_all_inventory,
    CUSTOMER_SCENARIOS,
    CUSTOMER_SCENARIO_LABELS,
)

logger = logging.getLogger("coffee_shop.notebook_ui")

AGENT_CONFIG = {
    'order_agent': {
        'icon': '\U0001f4dd',
        'name': 'Order Agent',
        'color': '#2196F3',
        'bg_color': '#E3F2FD'
    },
    'inventory_agent': {
        'icon': '\U0001f4e6',
        'name': 'Inventory Agent',
        'color': '#FF9800',
        'bg_color': '#FFF3E0'
    },
    'barista_agent': {
        'icon': '☕',
        'name': 'Barista Agent',
        'color': '#8BC34A',
        'bg_color': '#F1F8E9'
    },
    'customer_service_agent': {
        'icon': '\U0001f4ac',
        'name': 'Customer Service',
        'color': '#E91E63',
        'bg_color': '#FCE4EC'
    },
    'user': {
        'icon': '\U0001f464',
        'name': 'You',
        'color': '#424242',
        'bg_color': '#F5F5F5'
    }
}


class NotebookUI:
    """Jupyter ipywidgets interface for the coffee shop."""

    def __init__(self, app, customer_agent, mlflow_enabled=True, setup_name: str | None = None):
        self.app = app
        self.customer_agent = customer_agent
        self.mlflow_enabled = mlflow_enabled
        self.setup_name = setup_name
        self.agent_config = AGENT_CONFIG
        self.traces_of_latest_conversations = []
        self.verbose_mode = True
        self.customer_agent_enabled = False
        self._last_agent_message = None

    def _get_config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}

    def _format_content_for_display(self, content):
        content = normalize_content(content)
        try:
            parsed_json = json.loads(content)
            formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            return f'<div class="tool-output"><div class="tool-output-label">Output:</div><pre class="tool-output-code">{html.escape(formatted_json)}</pre></div>'
        except (json.JSONDecodeError, TypeError):
            pass
        return html.escape(content)

    def _should_show_message_in_silent_mode(self, agent_name, content):
        return agent_name in self.agent_config

    def _format_message_bubble(self, agent_name, content, is_user=False, is_important=True):
        if is_user:
            config = self.agent_config.get('user')
        else:
            config = self.agent_config.get(agent_name, {
                'icon': '\U0001f916',
                'name': "Uses tool: " + agent_name.replace('_', ' ').title(),
                'color': '#666666',
                'bg_color': '#F0F0F0'
            })

        formatted_content = self._format_content_for_display(content)

        bubble_html = f"""
        <div style="
            margin: 10px 0;
            padding: 0;
            display: flex;
            align-items: flex-start;
            {'justify-content: flex-end;' if is_user else 'justify-content: flex-start;'}
        " class="chat-bubble {'chat-verbose-message' if not is_important else ''}" >
            <div style="
                max-width: 70%;
                background-color: {config['bg_color']};
                background: linear-gradient(135deg, {config['bg_color']}44, {config['bg_color']}ff);
                border: 2px solid {config['color']};
                border-radius: 15px;
                padding: 12px 16px;
                margin: 0 10px;
                position: relative;
                {'order: 1;' if is_user else ''}
            ">
                <div style="
                    font-weight: bold;
                    color: {config['color']};
                    font-size: 12px;
                    margin-bottom: 5px;
                    display: flex;
                    align-items: center;
                    gap: 5px;
                ">
                    <span style="font-size: 16px;">{config['icon']}</span>
                    {config['name']}
                </div>
                <div style="
                    color: #333;
                    line-height: 1.4;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                ">{formatted_content}</div>
            </div>
        </div>
        """
        return bubble_html

    def _auto_scroll_to_bottom(self):
        scroll_script = """
        <script>
        (function() {
            var outputs = document.querySelectorAll('.widget-output, .jp-OutputArea-output');
            for (var i = 0; i < outputs.length; i++) {
                var output = outputs[i];
                if (output.style.height === '500px' || output.classList.contains('chat-output')) {
                    output.scrollTop = output.scrollHeight;
                    break;
                }
            }
            var chatOutputs = document.querySelectorAll('.chat-output');
            chatOutputs.forEach(function(element) {
                element.scrollTop = element.scrollHeight;
            });
            var scrollContainers = document.querySelectorAll('[style*="overflow: auto"], [style*="overflow:auto"]');
            scrollContainers.forEach(function(container) {
                if (container.style.height === '500px') {
                    container.scrollTop = container.scrollHeight;
                }
            });
        })();
        </script>
        """
        display(HTML(scroll_script))

    def _stream_to_output(self, stream, output_widget):
        with output_widget:
            for sm in extract_messages(stream):
                agent_name = sm.agent_name
                content = sm.content
                if sm.is_agent_reply:
                    self._last_agent_message = content
                is_important = self._should_show_message_in_silent_mode(agent_name, content)
                bubble_html = self._format_message_bubble(agent_name, content, is_user=False, is_important=is_important)
                display(HTML(bubble_html))
                self._auto_scroll_to_bottom()

    def _set_processing_status(self, is_processing=True):
        if hasattr(self, 'status_indicator'):
            if is_processing:
                self.status_indicator.value = "⏳ Agents are working on your request..."
                self.text_input.disabled = True
                self.send_button.disabled = True
                if hasattr(self, 'restock_button'):
                    self.restock_button.disabled = True
                for button in self.scenario_buttons.children:
                    button.disabled = True
            else:
                self.status_indicator.value = ""
                if not self.customer_agent_enabled:
                    self.text_input.disabled = False
                    self.send_button.disabled = False
                    self.text_input.focus()
                if hasattr(self, 'restock_button'):
                    self.restock_button.disabled = False
                for button in self.scenario_buttons.children:
                    button.disabled = False

    def continue_conversation_interactive(self, thread_id, prompt, output_widget):
        config = self._get_config(thread_id)
        self._set_processing_status(True)

        with output_widget:
            user_bubble = self._format_message_bubble('user', prompt, is_user=True, is_important=True)
            display(HTML(user_bubble))

        try:
            self._stream_to_output(
                self.app.stream(
                    {"messages": [{"role": "user", "content": prompt}], "handoff_context": None},
                    config,
                    subgraphs=True,
                ),
                output_widget
            )
            if self.mlflow_enabled:
                trace_id = mlflow.get_last_active_trace_id()
                self.traces_of_latest_conversations.append(trace_id)
                if trace_id is not None and self.setup_name is not None:
                    _tag_trace(trace_id, self.setup_name, -1)

                with output_widget:
                    display(HTML(f"""
                    <div style="
                        font-size: 10px;
                        color: #999;
                        text-align: center;
                        margin: 10px 0;
                        padding: 5px;
                        border-top: 1px solid #eee;
                    ">
                        Trace ID: {trace_id}
                    </div>
                    """))

        finally:
            self._set_processing_status(False)

        if self.customer_agent_enabled and self._last_agent_message:
            next_msg = self.customer_agent.respond_to(self._last_agent_message)
            self._last_agent_message = None
            if next_msg:
                self.continue_conversation_interactive(thread_id, next_msg, output_widget)
            else:
                with output_widget:
                    display(HTML("""
                    <div style="
                        background: linear-gradient(45deg, #e8f5e9, #a5d6a7);
                        border: 1px solid #4caf50;
                        border-radius: 10px;
                        padding: 12px;
                        margin: 10px 0;
                        text-align: center;
                        color: #1b5e20;
                    ">
                        \U0001f916 Auto Customer conversation complete.
                    </div>
                    """))

    def _inject_enhanced_css(self):
        css_style = f"<style>\n{ENHANCED_CSS}\n</style>"
        return widgets.HTML(css_style)

    def create_interactive_interface(self, success_only=False):
        self.current_thread_id = None

        css_widget = self._inject_enhanced_css()

        self.output = widgets.Output()
        self.output.add_class('chat-output')

        self.text_input = widgets.Text(
            value='',
            placeholder='Type your message to the coffee shop here...',
            description='Your Message:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%', height='35px')
        )
        self.text_input.add_class('default-input')
        self.text_input.on_submit(self._on_text_submit)

        self.send_button = widgets.Button(
            description='Send \U0001f4e4',
            button_style='primary',
            layout=widgets.Layout(width="120px", height='35px'),
            tooltip='Send your message'
        )
        self.send_button.add_class('default-button')
        self.send_button.on_click(self._on_send_button_clicked)

        self.status_indicator = widgets.HTML(value="")
        self.status_indicator.add_class('status-indicator')

        input_line = widgets.HBox([self.text_input, self.send_button])
        input_line.add_class('input-line')

        input_area = widgets.HBox([
            input_line,
            self.status_indicator,
        ], layout=widgets.Layout(justify_content='flex-start', align_items='center'))
        input_area.add_class('input-area')

        controls_header = widgets.HTML("""
        <div>
            <h4 style="margin: 0; color: #007bff;">Chat Control</h4>
            <p style="margin: 0; color: #6c757d;">Use the buttons below to overall control the chat.</p>
        </div>
        """)

        self.new_conversation_button = widgets.Button(
            description='\U0001f195 New Chat',
            button_style='info',
            tooltip='Start a new conversation'
        )
        self.new_conversation_button.add_class('default-button')
        self.new_conversation_button.on_click(self._on_new_conversation_clicked)

        self.customer_agent_toggle = widgets.ToggleButton(
            value=self.customer_agent_enabled,
            description='\U0001f916 Auto Customer: Off',
            disabled=False,
            button_style='',
            tooltip='Toggle automatic customer agent that guides the conversation'
        )
        self.customer_agent_toggle.add_class('default-button')
        self.customer_agent_toggle.observe(self._on_customer_agent_toggle_changed, names='value')

        self.customer_scenario_dropdown = widgets.Dropdown(
            options=[(f'Scenario {i+1}: {CUSTOMER_SCENARIO_LABELS[i]}', i) for i in range(len(CUSTOMER_SCENARIOS))],
            value=0,
            description='Scenario:',
            style={'description_width': '65px'},
            layout=widgets.Layout(width='320px'),
            disabled=True,
        )
        self.customer_scenario_dropdown.observe(self._on_customer_scenario_changed, names='value')

        self.verbose_toggle = widgets.ToggleButton(
            value=self.verbose_mode,
            description='\U0001f50a Verbose: On',
            disabled=False,
            button_style='info',
            tooltip='Toggle between verbose (show all messages) and silent (hide tool calls) modes'
        )
        self.verbose_toggle.add_class('default-button')
        self.verbose_toggle.observe(self._on_verbose_toggle_changed, names='value')

        self.log_level_dropdown = widgets.Dropdown(
            options=[
                ('Debug', logging.DEBUG),
                ('Info', logging.INFO),
                ('Warning', logging.WARNING),
                ('Error', logging.ERROR),
            ],
            value=logging.INFO,
            description='Log Level:',
            style={'description_width': '65px'},
            layout=widgets.Layout(width='180px'),
        )
        self.log_level_dropdown.observe(self._on_log_level_changed, names='value')

        self.restock_button = widgets.Button(
            description='\U0001f504 Restock All Items',
            button_style='success',
            tooltip='Restock all items to full inventory'
        )
        self.restock_button.add_class('default-button')
        self.restock_button.on_click(self._on_restock_clicked)

        controls_buttons = widgets.HBox([
            self.new_conversation_button,
            self.verbose_toggle,
            self.restock_button,
            self.customer_agent_toggle,
            self.log_level_dropdown,
            self.customer_scenario_dropdown,
        ])
        controls_buttons.add_class('button-group')

        controls_area = widgets.VBox([controls_header, controls_buttons])
        controls_area.add_class('scenario-area')

        scenario_header = widgets.HTML("""
        <div>
            <h4 style="margin: 0; color: #007bff;">Quick Start Scenarios</h4>
            <p style="margin: 0; color: #6c757d;">These buttons will initiate a new conversation using a predefined message.</p>
        </div>
        """)

        scenario_buttons = []
        if success_only:
            scenario_buttons.append(widgets.Button(
                description='\U0001f6cd️ Successful Order',
                button_style='success',
                tooltip='Order 2 lattes and a croissant'
            ))
        else:
            scenario_buttons.append(widgets.Button(
                description='❓ Menu Issue',
                button_style='warning',
                tooltip='Order item not on menu'
            ))
            scenario_buttons.append(widgets.Button(
                description='\U0001f4e6 Inventory Issue',
                button_style='danger',
                tooltip='Order item out of stock'
            ))
            scenario_buttons.append(widgets.Button(
                description='\U0001f61e Complaint',
                button_style='info',
                tooltip='Complain about a drink'
            ))

        for button in scenario_buttons:
            button.add_class('scenario-button')
            button.add_class('default-button')

        self.scenario_buttons = widgets.HBox(scenario_buttons)
        self.scenario_buttons.add_class('button-group')

        for i, button in enumerate(self.scenario_buttons.children):
            button.on_click(lambda b, scenario=i: self._on_scenario_clicked(b, scenario))

        scenario_area = widgets.VBox([scenario_header, self.scenario_buttons])
        scenario_area.add_class('scenario-area')

        controls = widgets.HBox([scenario_area, controls_area])
        controls.add_class('controls-container')

        chat_area = widgets.VBox([self.output, input_area])
        chat_area.add_class('chat-area')

        interface = widgets.VBox([
            css_widget,
            widgets.HTML('<div style="margin: 5px 0;"></div>'),
            chat_area,
            controls
        ])
        interface.add_class('chat-container')

        self._start_new_conversation()

        return interface

    def _on_send_button_clicked(self, button):
        message = self.text_input.value.strip()
        if message:
            self.continue_conversation_interactive(self.current_thread_id, message, self.output)
            self.text_input.value = ''

    def _on_text_submit(self, text_widget):
        self._on_send_button_clicked(None)

    def _on_new_conversation_clicked(self, button):
        print("These are the trace IDs of the latest conversations in this session:")
        for trace_id in self.traces_of_latest_conversations:
            print(f"- {trace_id}")
        self.traces_of_latest_conversations = []
        self._start_new_conversation()

    def _start_new_conversation(self):
        self.current_thread_id = str(uuid.uuid4())
        self._last_agent_message = None

        with self.output:
            clear_output()
            welcome_html = """
            <div style="
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 15px;
            ">
                <h3 style="margin: 0;">\U0001f195 New Conversation Started!</h3>
            </div>
            """
            display(HTML(welcome_html))

        if self.customer_agent_enabled and self.customer_agent:
            scenario_idx = self.customer_scenario_dropdown.value if hasattr(self, 'customer_scenario_dropdown') else None
            self.customer_agent.reset(scenario_idx)
            first_msg = self.customer_agent.get_initial_message()
            self.text_input.value = first_msg
            self._on_send_button_clicked(None)

    def _on_restock_clicked(self, button):
        reset_inventory()
        with self.output:
            restock_html = """
            <div style="
                background: linear-gradient(45deg, #d4edda, #a3d977);
                border: 1px solid #28a745;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                text-align: center;
            ">
                <h4 style="margin: 0 0 10px 0; color: #155724;">\U0001f504 Inventory Successfully Restocked!</h4>
            </div>
            """
            display(HTML(restock_html))
        self.display_current_inventory()

    def display_current_inventory(self):
        with self.output:
            inventory_html = '<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">'
            inventory_html += '<h5 style="margin: 0 0 10px 0; color: #495057;">\U0001f4e6 Current Inventory Levels:</h5>'
            inventory_html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">'

            for item_key, item in get_all_inventory().items():
                inventory_html += f"""
                <div style="
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 3px solid #007bff;
                ">
                    <strong>{item.name}</strong><br>
                    <span style="color: #28a745;">Stock: {item.stock} units</span><br>
                    <span style="color: #6c757d;">${item.price:.2f}</span>
                </div>
                """

            inventory_html += '</div></div>'
            display(HTML(inventory_html))

    def _on_scenario_clicked(self, button, scenario):
        self._start_new_conversation()

        scenarios = []
        if len(self.scenario_buttons.children) == 1:
            scenarios = [
                "I'd like to order 2 large lattes with almond milk and 1 croissant please"
            ]
        else:
            scenarios = [
                "I want 1 croissant and 1 piece of cheesecake",
                "Can I get 2 muffins please?",
                "I'm not happy with my latte, it tastes bitter and wrong"
            ]
            if scenario == 1:
                set_item_stock('muffin', 0)
                with self.output:
                    restock_html = """
                    <div class="chat-notification">
                        <h4>All muffins vanished from the inventory!</h4>
                    </div>
                    """
                    display(HTML(restock_html))
                self.display_current_inventory()
            else:
                inventory = get_all_inventory()
                if inventory.get('muffin') and inventory['muffin'].stock == 0:
                    with self.output:
                        restock_html = """
                        <div class="chat-notification">
                            <h4>Fresh muffins arrived!</h4>
                        </div>
                        """
                        display(HTML(restock_html))
                    set_item_stock('muffin', 12)

        self.text_input.value = scenarios[scenario]
        self._on_send_button_clicked(None)

    def _on_log_level_changed(self, change):
        logging.getLogger("coffee_shop").setLevel(change['new'])

    def _on_verbose_toggle_changed(self, change):
        self.verbose_mode = change['new']
        if self.verbose_mode:
            self.output.remove_class('chat-silent-mode')
            self.verbose_toggle.description = '\U0001f50a Verbose: On'
            self.verbose_toggle.button_style = 'info'
        else:
            self.verbose_toggle.description = '\U0001f507 Verbose: Off'
            self.verbose_toggle.button_style = 'warning'
            self.output.add_class('chat-silent-mode')

    def _on_customer_agent_toggle_changed(self, change):
        self.customer_agent_enabled = change['new']
        if self.customer_agent_enabled:
            self.customer_agent_toggle.description = '\U0001f916 Auto Customer: On'
            self.customer_agent_toggle.button_style = 'success'
            self.customer_scenario_dropdown.disabled = False
            self.text_input.disabled = True
            self.send_button.disabled = True
            self._start_new_conversation()
        else:
            self.customer_agent_toggle.description = '\U0001f916 Auto Customer: Off'
            self.customer_agent_toggle.button_style = ''
            self.customer_scenario_dropdown.disabled = True
            self.text_input.disabled = False
            self.send_button.disabled = False

    def _on_customer_scenario_changed(self, change):
        if self.customer_agent and self.customer_agent_enabled:
            self._start_new_conversation()
