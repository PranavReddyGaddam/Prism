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
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Textarea } from "@/components/ui/textarea";
import { useAutoResizeTextarea } from "@/hooks/use-auto-resize-textarea";
import { cn } from "@/lib/utils";


export default function AI_Prompt({ onSubmit }: { onSubmit?: (prompt: string) => void }) {
  const [value, setValue] = useState("");
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 72,
    maxHeight: 300,
  });

  const COT_QUESTIONS = [
    "A father is 4 times as old as his son. In 20 years, he will be twice as old as his son. How old is the son now?",
    "There are chickens and cows in a farm. There are 30 heads and 84 legs in total. How many chickens and how many cows are there?",
    "A bag contains 5 red balls and 7 blue balls. Two balls are drawn without replacement. What is the probability that both balls are red?"
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
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        className="flex h-8 items-center gap-1 rounded-md pr-2 pl-1 text-xs text-white hover:bg-gray-900 focus-visible:ring-1 focus-visible:ring-blue-500 focus-visible:ring-offset-0"
                        variant="ghost"
                      >
                        <span>Question</span>
                        <ChevronDown className="h-3 w-3" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="bg-black border-gray-800">
                      {COT_QUESTIONS.map((question) => (
                        <DropdownMenuItem
                          key={question}
                          className="text-white hover:bg-gray-900 focus:bg-gray-900 cursor-pointer"
                          onClick={() => setValue(question)}
                        >
                          <span className="text-sm">{question}</span>
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
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
