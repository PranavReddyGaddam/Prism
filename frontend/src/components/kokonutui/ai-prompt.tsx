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

import { ArrowRight, Bot, Check, ChevronDown } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
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


export default function AI_Prompt({ onSubmit }: { onSubmit?: (prompt: string, model: string) => void }) {
  const [value, setValue] = useState("");
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 72,
    maxHeight: 300,
  });
  const [selectedModel, setSelectedModel] = useState("DeepSeek");

  const COT_QUESTIONS = [
    "A father is 4 times as old as his son. In 20 years, he will be twice as old as his son. How old is the son now?",
    "There are chickens and cows in a farm. There are 30 heads and 84 legs in total. How many chickens and how many cows are there?",
    "A bag contains 5 red balls and 7 blue balls. Two balls are drawn without replacement. What is the probability that both balls are red?"
  ];

  const AI_MODELS = [
    "Microsoft Phi",
    "DeepSeek",
  ];

  const MODEL_ICONS: Record<string, React.ReactNode> = {
    "Microsoft Phi": (
      <svg height="1em" style={{ flex: "none", lineHeight: "1" }} viewBox="0 0 24 24" width="1em" xmlns="http://www.w3.org/2000/svg">
        <title>Microsoft</title>
        <path d="M11.4 11.4H0V0h11.4v11.4zM24 11.4H12.6V0H24v11.4zM11.4 24H0V12.6h11.4V24zM24 24H12.6V12.6H24V24z" fill="currentColor"/>
      </svg>
    ),
    "DeepSeek": (
      <svg fill="currentColor" fillRule="evenodd" height="1em" style={{ flex: "none", lineHeight: "1" }} viewBox="0 0 24 24" width="1em" xmlns="http://www.w3.org/2000/svg">
        <title>DeepSeek</title>
        <path d="M23.748 4.482c-.254-.124-.364.113-.512.234-.051.039-.094.09-.137.136-.372.397-.806.657-1.373.626-.829-.046-1.537.214-2.163.848-.133-.782-.575-1.248-1.247-1.548-.352-.156-.708-.311-.955-.65-.172-.241-.219-.51-.305-.774-.055-.16-.11-.323-.293-.35-.2-.031-.278.136-.356.276-.313.572-.434 1.202-.422 1.84.027 1.436.633 2.58 1.838 3.393.137.093.172.187.129.323-.082.28-.18.552-.266.833-.055.179-.137.217-.329.14a5.526 5.526 0 01-1.736-1.18c-.857-.828-1.631-1.742-2.597-2.458a11.365 11.365 0 00-.689-.471c-.985-.957.13-1.743.388-1.836.27-.098.093-.432-.779-.428-.872.004-1.67.295-2.687.684a3.055 3.055 0 01-.465.137 9.597 9.597 0 00-2.883-.102c-1.885.21-3.39 1.102-4.497 2.623C.082 8.606-.231 10.684.152 12.85c.403 2.284 1.569 4.175 3.36 5.653 1.858 1.533 3.997 2.284 6.438 2.14 1.482-.085 3.133-.284 4.994-1.86.47.234.962.327 1.78.397.63.059 1.236-.03 1.705-.128.735-.156.684-.837.419-.961-2.155-1.004-1.682-.595-2.113-.926 1.096-1.296 2.746-2.642 3.392-7.003.05-.347.007-.565 0-.845-.004-.17.035-.237.23-.256a4.173 4.173 0 001.545-.475c1.396-.763 1.96-2.015 2.093-3.517.02-.23-.004-.467-.247-.588zM11.581 18c-2.089-1.642-3.102-2.183-3.52-2.16-.392.024-.321.471-.235.763.09.288.207.486.371.739.114.167.192.416-.113.603-.673.416-1.842-.14-1.897-.167-1.361-.802-2.5-1.86-3.301-3.307-.774-1.393-1.224-2.887-1.298-4.482-.02-.386.093-.522.477-.592a4.696 4.696 0 011.529-.039c2.132.312 3.946 1.265 5.468 2.774.868.86 1.525 1.887 2.202 2.891.72 1.066 1.494 2.082 2.48 2.914.348.292.625.514.891.677-.802.09-2.14.11-3.054-.614zm1-6.44a.306.306 0 01.415-.287.302.302 0 01.2.288.306.306 0 01-.31.307.303.303 0 01-.304-.308zm3.11 1.596c-.2.081-.399.151-.59.16a1.245 1.245 0 01-.798-.254c-.274-.23-.47-.358-.552-.758a1.73 1.73 0 01.016-.588c.07-.327-.008-.537-.239-.727-.187-.156-.426-.199-.688-.199a.559.559 0 01-.254-.078c-.11-.07-.16-.18-.16-.295 0-.117.05-.227.16-.295a.559.559 0 01.254-.078c.262 0 .501.043.688.199.231.19.31.4.239.727-.082.4.278.528.552.758.274.23.52.329.798.254.191-.009.39-.079.59-.16.2-.082.399-.152.59-.16a1.245 1.245 0 01.798.254c.274.23.47.358.552.758a1.73 1.73 0 01-.016.588c-.07.327.008.537.239.727.187.156.426.199.688.199a.559.559 0 01.254.078c.11.07.16.18.16.295 0 .117-.05.227-.16.295a.559.559 0 01-.254.078c-.262 0-.501-.043-.688-.199-.231-.19-.31-.4-.239-.727.082-.4-.278-.528-.552-.758-.274-.23-.52-.329-.798-.254-.191.009-.39.079-.59.16z"/>
      </svg>
    ),
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (value.trim() && onSubmit) {
      onSubmit(value.trim(), selectedModel);
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
                        <AnimatePresence mode="wait">
                          <motion.div
                            animate={{
                              opacity: 1,
                              y: 0,
                            }}
                            className="flex items-center gap-1"
                            exit={{
                              opacity: 0,
                              y: 5,
                            }}
                            initial={{
                              opacity: 0,
                              y: -5,
                            }}
                            key={selectedModel}
                            transition={{
                              duration: 0.15,
                            }}
                          >
                            {MODEL_ICONS[selectedModel]}
                            {selectedModel}
                            <ChevronDown className="h-3 w-3 opacity-50" />
                          </motion.div>
                        </AnimatePresence>
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent
                      className={cn(
                        "min-w-[10rem]",
                        "bg-black border-gray-800"
                      )}
                    >
                      {AI_MODELS.map((model) => (
                        <DropdownMenuItem
                          className="flex items-center justify-between gap-2 text-white hover:bg-gray-900 focus:bg-gray-900"
                          key={model}
                          onSelect={() => setSelectedModel(model)}
                        >
                          <div className="flex items-center gap-2">
                            {MODEL_ICONS[model] || (
                              <Bot className="h-4 w-4 opacity-50" />
                            )}{" "}
                            {/* Use mapped SVG or fallback */}
                            <span>{model}</span>
                          </div>
                          {selectedModel === model && (
                            <Check className="h-4 w-4 text-blue-500" />
                          )}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <div className="mx-0.5 h-4 w-px bg-gray-700" />
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
