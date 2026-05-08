"use client";

/**
 * @author: @kokonutui
 * @description: AI Prompt Input
 * @version: 1.0.0
 * @date: 2025-06-26
 * @license: MIT
 * @website: https://kokonutui.com
 * @github: https://github.com/kokonut-labs/kokonutui
 */

import { ArrowRight, ChevronDown } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useAutoResizeTextarea } from "@/hooks/use-auto-resize-textarea";
import { cn } from "@/lib/utils";


export default function AI_Prompt({ onSubmit }: { onSubmit?: (prompt: string) => void }) {
  const [value, setValue] = useState("");
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 72,
    maxHeight: 300,
  });

  const COT_QUESTIONS = [
    // Original
    "A father is 4 times as old as his son. In 20 years, he will be twice as old as his son. How old is the son now?",
    "There are chickens and cows in a farm. There are 30 heads and 84 legs in total. How many chickens and how many cows are there?",
    "A bag contains 5 red balls and 7 blue balls. Two balls are drawn without replacement. What is the probability that both balls are red?",
    // New
    "A train travels 120 km at 60 km/h, then 180 km at 90 km/h. What is the average speed for the entire journey?",
    "If 3x + 7 = 22, find the value of 5x - 3.",
    "A store sells apples for $1.20 each and oranges for $0.80 each. If John buys 5 apples and 8 oranges and pays with a $20 bill, how much change does he get?",
    "Two pipes can fill a tank in 6 hours and 8 hours respectively. If both are opened together, how long will it take to fill the tank?",
    "What is the remainder when 2^100 is divided by 7?",
    "Find the greatest common divisor of 252 and 198.",
    "A circle has an area of 49π. What is its circumference?",
    "In a class of 30 students, 18 play football and 15 play basketball. If 10 play both, how many play neither?",
  ];

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (value.trim() && onSubmit) {
      onSubmit(value.trim());
      setValue("");
      adjustHeight(true);
    }
  };

  return (
    <div className="w-full py-4">
      <div className="rounded-2xl bg-black border border-gray-800 p-1.5 pt-4">
                <div className="relative">
          <div className="relative flex flex-col">
            <div className="overflow-y-auto" style={{ maxHeight: "400px" }}>
              <Textarea
                className={cn(
                  "w-full resize-none rounded-xl rounded-b-none border-none bg-black px-4 py-3 placeholder:text-gray-500 focus-visible:ring-0 focus-visible:ring-offset-0 text-white",
                  "min-h-[72px]"
                )}
                id="ai-input-15"
                onChange={(e) => {
                  setValue(e.target.value);
                  adjustHeight();
                }}
                onKeyDown={handleKeyDown}
                placeholder={"Enter a Query to Analyze..."}
                ref={textareaRef}
                value={value}
              />
            </div>

            <div className="flex h-14 items-center rounded-b-xl bg-black border-t border-gray-800">
              <div className="absolute right-3 bottom-3 left-3 flex w-[calc(100%-24px)] items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="relative" ref={dropdownRef}>
                    <Button
                      className="flex h-8 items-center gap-1.5 rounded-md px-2.5 text-xs text-gray-400 hover:text-white hover:bg-white/5 focus-visible:ring-0 border border-transparent hover:border-white/10 transition-all"
                      variant="ghost"
                      onClick={() => setOpen(o => !o)}
                      type="button"
                    >
                      <span>Sample question</span>
                      <ChevronDown className={cn("h-3 w-3 transition-transform duration-150", open && "rotate-180")} />
                    </Button>
                    {open && (
                      <div
                        className="absolute bottom-full left-0 mb-2 z-50 rounded-xl overflow-hidden"
                        style={{
                          width: '480px',
                          backgroundColor: '#141414',
                          border: '1px solid rgba(255,255,255,0.08)',
                          boxShadow: '0 -8px 32px rgba(0,0,0,0.6)',
                        }}
                      >
                        <div className="px-3 py-2" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                          <span className="text-[10px] uppercase tracking-widest" style={{ color: '#6b7280' }}>Sample questions</span>
                        </div>
                        <div className="overflow-y-auto" style={{ maxHeight: '280px' }}>
                          {COT_QUESTIONS.map((question, i) => (
                            <button
                              key={i}
                              type="button"
                              className="w-full text-left px-3 py-2.5 text-sm transition-colors"
                              style={{ color: '#d1d5db', borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                              onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)')}
                              onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
                              onClick={() => { setValue(question); setOpen(false); adjustHeight(); }}
                            >
                              <span className="text-[10px] font-mono mr-2" style={{ color: '#4b5563' }}>{String(i + 1).padStart(2, '0')}</span>
                              {question}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                <button
                  aria-label="Send message"
                  className={cn(
                    "rounded-lg bg-gray-900 p-2",
                    "hover:bg-gray-800 focus-visible:ring-1 focus-visible:ring-blue-500 focus-visible:ring-offset-0"
                  )}
                  disabled={!value.trim()}
                  type="button"
                  onClick={handleSubmit}
                >
                  <ArrowRight
                    className={cn(
                      "h-4 w-4 transition-opacity duration-200 text-white",
                      value.trim() ? "opacity-100" : "opacity-30"
                    )}
                  />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
