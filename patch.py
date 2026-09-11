import re

with open("app/frontend/src/App.jsx", "r", encoding="utf-8") as f:
    content = f.read()

# Rebrand Name
content = content.replace("TrustDesk AI", "HiverSupport AI")

# Change Color Theme: indigo -> yellow
content = content.replace("indigo", "yellow")

# Add missing Lucide icons for new features
content = content.replace(
    "Scale\n} from 'lucide-react';",
    "Scale,\n  Trash2,\n  Copy,\n  Moon,\n  Sun\n} from 'lucide-react';"
)

# Add Dark Mode State
content = content.replace(
    "const [error, setError] = useState(null);",
    "const [error, setError] = useState(null);\n  const [isDarkMode, setIsDarkMode] = useState(true);"
)

# Apply Dark Mode to root container
content = content.replace(
    '<div className="flex h-screen bg-slate-950 text-slate-100 antialiased overflow-hidden">',
    '<div className={`flex h-screen ${isDarkMode ? "bg-slate-950 text-slate-100" : "bg-slate-50 text-slate-900"} antialiased overflow-hidden`}>'
)

# Add Dark Mode Toggle & Branding
content = content.replace(
    '<div className="p-6 border-b border-slate-800">',
    '''<div className="p-6 border-b border-slate-800">
            <div className="flex justify-end mb-2">
              <button onClick={() => setIsDarkMode(!isDarkMode)} className="p-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-yellow-400 transition flex items-center gap-2 text-xs">
                {isDarkMode ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
                {isDarkMode ? 'Light Mode' : 'Dark Mode'}
              </button>
            </div>'''
)

# Add Clear Session Button next to Evaluate Message
content = content.replace(
    '<button\n                  onClick={() => handlePredict(customerMessage)}',
    '''<button 
                  onClick={() => { setCustomerMessage(""); setPrediction(null); }}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold flex items-center gap-1.5 border border-slate-700 transition"
                >
                  <Trash2 className="w-3.5 h-3.5" /> Clear
                </button>
                <button
                  onClick={() => handlePredict(customerMessage)}'''
)

# Add Copy Response Button
content = content.replace(
    '{prediction.draft_reply}\n                    </div>',
    '''{prediction.draft_reply}
                    </div>
                    <div className="mt-3 flex justify-end">
                      <button 
                        onClick={() => navigator.clipboard.writeText(prediction.draft_reply)}
                        className="px-3 py-1.5 bg-yellow-600 hover:bg-yellow-500 text-slate-900 rounded-lg text-xs font-bold flex items-center gap-1.5 shadow-md transition"
                      >
                        <Copy className="w-3.5 h-3.5" /> Copy Response
                      </button>
                    </div>'''
)

with open("app/frontend/src/App.jsx", "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied successfully.")
