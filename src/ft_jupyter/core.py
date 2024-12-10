from __future__ import annotations
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Any, List, Dict, Optional
from contextlib import contextmanager
from fasthtml.common import *
from fasthtml.jupyter import *
from IPython.display import display, HTML 

def requires_shell(func: Callable) -> Callable[..., Optional[Any]]:
    """Decorator to ensure shell is available"""
    @wraps(func)
    def wrapper(self: 'NotebookContext', *args, **kwargs) -> Optional[Any]:
        return None if not self._shell else func(self, *args, **kwargs)
    return wrapper

@dataclass
class NotebookContext:
    """Manages IPython/Jupyter notebook context and variable tracking"""
    pages: List['Page'] = field(default_factory=list)
    _shell: Any = field(init=False, default=None)
    
    def __post_init__(self) -> None:
        """Initialize IPython shell connection"""
        from IPython import get_ipython
        self._shell = get_ipython()    
    
    @contextmanager 
    @requires_shell
    def show_ft(self):
        """Context manager for converting notebook code to FastHTML format."""
        class CodeCapture:
            def __init__(self):
                self.code = None    # For storing the code string
                self.content = None # For storing the evaluated content

        result = CodeCapture()
        yield result
                
        ip = self._shell
                
        cell = ip.history_manager.input_hist_parsed[-1]        
        
        # First get all lines
        all_lines = cell.splitlines()
        
        # Find the actual content (skip the with show_ft line)
        try:
            content_start = next(i for i, line in enumerate(all_lines) if 'show_ft()' in line) + 1
            code_lines = all_lines[content_start:]
        except StopIteration:
            code_lines = all_lines
        
        # Remove leading whitespace
        if code_lines:
            base_indent = len(code_lines[0]) - len(code_lines[0].lstrip())
            code_lines = [line[base_indent:] if line.strip() else line for line in code_lines]    
        
        # Check if this is a route handler
        if any("@manager.post" in line or "@manager.get" in line for line in code_lines):
            fasthtml_code = []
            for i, line in enumerate(code_lines):
                if "@manager." in line:
                    method = "get" if "get" in line else "post"
                    route = line.split('"')[1]
                    if not route.startswith('/'):
                        route = '/' + route
                    
                    func_def = code_lines[i + 1]
                    func_body = code_lines[i + 2:]
                    
                    base_indent = len(func_body[0]) - len(func_body[0].lstrip())
                    
                    fasthtml_code.append(f"@rt('{route}')\ndef {method}():")
                    for line in func_body:
                        if line.strip():
                            adjusted_line = "    " + line[base_indent:]
                            fasthtml_code.append(adjusted_line)
                        else:
                            fasthtml_code.append(line)
                    break
            
            result.code = "\n".join(fasthtml_code)
            
        elif any("create_page" in line for line in code_lines):
            fasthtml_code = []
            route = None
            for line in code_lines:
                if "create_page" in line:
                    route = line.split('"')[1]
                    break
            
            if route is not None:
                fasthtml_code.append(f"@rt('/{route}')\ndef get():")
                inside_add = False
                content_lines = []
                for line in code_lines:
                    if ".add(" in line:
                        inside_add = True
                        content_lines.append(line[line.find(".add(") + 5:].strip())
                    elif inside_add:
                        if line.strip().endswith(")"):
                            inside_add = False
                            content_lines.append(line.strip()[:-1])
                        else:
                            content_lines.append(line.strip())
                
                content = "\n        ".join(content_lines)
                fasthtml_code.append(f"    return Div(\n        {content}\n    )")
                
                result.code = "\n".join(fasthtml_code)
                
        else:
            # For UI code, capture both the code string and evaluate the content
            code_str = "\n".join(code_lines)
            result.code = code_str
            try:
                result.content = eval(code_str, ip.user_ns)            
            except Exception as e:
                print(f"\nError evaluating content: {e}")

    def register_page(self, page: 'Page') -> None:
        """Register a page for variable tracking"""
        self.pages.append(page)
    
    @requires_shell
    def auto_update_cell(self, _=None) -> None:
        """Update tracked variables that changed in last cell execution"""
        for page in self.pages:
            self._check_page_variables(page)
    
    def _check_page_variables(self, page: 'Page') -> None:
        """Check for changes in a page's tracked variables"""
        # Look for refs of type 'var'
        var_refs = {
            name: ref['elem'] 
            for name, ref in page.refs.items() 
            if ref['type'] == 'var'
        }
        
        for var_name in list(var_refs.keys()):
            if var_name not in self._shell.user_ns:
                continue
                
            new_value = self._shell.user_ns[var_name]
            old_value = var_refs[var_name]
            
            if new_value is not old_value:
                page.update_element(new_value, var_name)
    
    @requires_shell
    def get_variable_name(self, value: Any) -> str | None:
        """Find variable name in notebook namespace"""
        user_vars = {
            name: val 
            for name, val in self._shell.user_ns.items() 
            if not name.startswith('_')
        }
        
        return next(
            (name for name, val in user_vars.items() if val is value), 
            None
        )
    
    @requires_shell
    def register_callback(self, callback: Callable) -> None:
        """Register a post-cell execution callback"""
        self._shell.events.register('post_run_cell', callback)        
    
    @requires_shell
    def unregister_callback(self, callback: Callable) -> None:
        """Unregister a post-cell execution callback"""
        self._shell.events.unregister('post_run_cell', callback)        
    
@dataclass
class Page:
    client: Any 
    context: NotebookContext
    name: str = ""
    elements: List = field(default_factory=list)
    refs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active: bool = True
    dirty: bool = False

    def _update_element_at(self, elem: Any, idx: int) -> None:
        """Update element at specific index and its references"""
        self.elements[idx] = elem
        
        # Update any existing refs pointing to this index
        for key, ref in list(self.refs.items()):
            if ref['index'] == idx:
                ref['elem'] = elem

    def _handle_element(self, elem: Any) -> None:
        """Add or update a single element"""
        elem_id = getattr(elem, 'id', None)
        var_name = self.context.get_variable_name(elem)
        
        # Find existing index from refs
        idx = next(
            (ref['index'] 
             for k in [elem_id, var_name] 
             if k and k in self.refs 
             and (ref := self.refs[k])),
            None
        )
        
        if idx is not None:
            self._update_element_at(elem, idx)
        else:
            idx = len(self.elements)
            self.elements.append(elem)
                    
        for key, type_ in [(var_name, 'var'), (elem_id, 'id')]:
            if key:
                self.refs[key] = {'type': type_, 'index': idx, 'elem': elem}
        
        self.dirty = True

    def add(self, *elements: Any) -> 'Page':
        """Add or update elements"""
        for elem in elements:
            self._handle_element(elem)
        return self.update()

    def update_element(self, elem: Any, var_name: str | None = None) -> None:
        """Update or remove element by variable name"""
        if var_name is None:
            var_name = self.context.get_variable_name(elem)
        
        if not var_name:
            return
            
        if elem is None:
            self.remove(var_name)
        else:
            self._handle_element(elem)
            self.update()

    def remove(self, *ref_keys: str) -> 'Page':
        """Remove elements by reference keys"""
        removed_indices = []
        
        for key in ref_keys:
            if ref := self.refs.get(key):
                removed_indices.append(ref['index'])
                del self.refs[key]
        
        # Remove elements in reverse order to maintain correct indices
        for idx in sorted(removed_indices, reverse=True):
            self.elements.pop(idx)
            # Update remaining refs
            for ref in self.refs.values():
                if ref['index'] > idx:
                    ref['index'] -= 1
        
        if removed_indices:
            self.dirty = True
            
        return self.update()

    def update(self) -> 'Page':
        """Update page content if active and dirty"""
        if self.active and self.dirty:
            self.client.set(*self.elements)
            self.dirty = False
        return self

    def clear(self) -> 'Page':
        """Clear all elements from the page"""
        self.elements.clear()
        self.refs.clear()
        self.dirty = True
        return self.update()
    
    def get_html(self) -> str:
        """Get HTML representation of current page"""
        if not self.elements:
            return ""
        from fasthtml.common import to_xml
        html = "\n".join(to_xml(elem) for elem in self.elements)
        print(html)

    def __repr__(self) -> str:
        route_name = self.name or "home"
        url = f"http://localhost:8000/{self.name}"
        return f"Page(route='{route_name}', elements={len(self.elements)}, url='{url}')"
    
@dataclass
class PageManager:
    """Manages FastHTML pages and websocket connections"""
    exts: str = "ws"
    pages: Dict[str, Page] = field(default_factory=dict)
    _common_elements: List = field(default_factory=list)
    _context: NotebookContext = field(init=False)
    _app: Any = field(init=False)
    _server: Any = field(init=False)
    _callback_registered: bool = field(init=False, default=False)
    
    def __post_init__(self) -> None:
        """Initialize FastHTML app and notebook context"""
        # Setup FastHTML        
        self._app = FastHTML(exts=self.exts)
        self._server = JupyUvi(self._app)
        setup_ws(self._app)
        
        # Setup notebook context
        self._context = NotebookContext()
        self._callback_registered = self._context.register_callback(
            self._context.auto_update_cell
        )

    def add_to_all(self, *elements: Any) -> None:
        """Add elements to all pages (existing and future)"""
        # Add to existing pages
        for page in self.pages.values():
            page.add(*elements)
        
        # Store for future pages
        self._common_elements.extend(elements)

    def create_page(self, route: str = "", frame: bool = False) -> Page:
        """Create a new page at the specified route"""
        client = ws_client(self._app, route, frame=frame, link=False)
        
        # Create and register page
        page = Page(client, self._context, route)
        self.pages[route] = page
        self._context.register_page(page)
        
        # Add common elements to the new page
        if self._common_elements:
            page.add(*self._common_elements)
        
        # Show page link
        self._display_page_link(route)
        return page    
    
    def _display_page_link(self, route: str) -> None:
        """Display clickable link to page"""
        route_name = route or "home"
        url = f"http://localhost:8000/{route}"
        display(HTML(f'<a href="{url}" target="_blank">View {route_name} page</a>'))
    
    def update_all(self) -> None:
        """Update all active pages that have changes"""
        for page in self.pages.values():
            if page.active and page.dirty:
                page.update()
    
    def stop(self) -> None:
        """Clean up manager resources"""
        if self._callback_registered:
            self._context.unregister_callback(self._context.auto_update_cell)
            self._callback_registered = False
        if hasattr(self, '_server'):
            self._server.stop()
    
    def __del__(self) -> None:
        """Ensure cleanup on deletion"""
        self.stop()
    
    def activate_route(self, route: str) -> None:
        """Enable auto-updates for a route"""
        if page := self.pages.get(route):
            page.active = True
    
    def deactivate_route(self, route: str) -> None:
        """Disable auto-updates for a route"""
        if page := self.pages.get(route):
            page.active = False

    def post(self, path: str):
        """Decorator for POST routes"""
        return self._app.post(path)
    
    def get(self, path: str):
        """Decorator for GET routes"""
        return self._app.get(path)
        
    @contextmanager
    def show_ft(self):
        """Convenience method to access show_ft from the context"""
        with self._context.show_ft() as code:
            yield code