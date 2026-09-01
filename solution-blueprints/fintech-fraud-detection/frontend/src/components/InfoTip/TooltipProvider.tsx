// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

type OpenMode = "hover" | "pinned";

interface OpenState {
  id: string;
  mode: OpenMode;
}

interface TooltipController {
  openId: string | null;
  openMode: OpenMode | null;
  open: (id: string, mode: OpenMode) => void;
  close: (id: string) => void;
  pin: (id: string) => void;
  registerPopup: (id: string, node: HTMLElement | null) => void;
}

const Ctx = createContext<TooltipController | null>(null);

export function useTooltipController(): TooltipController {
  const c = useContext(Ctx);
  if (!c) throw new Error("InfoTip must be used inside <TooltipProvider>");
  return c;
}

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<OpenState | null>(null);
  const popups = useRef<Map<string, HTMLElement>>(new Map());

  const open = useCallback((id: string, mode: OpenMode) => {
    setState({ id, mode });
  }, []);

  const close = useCallback((id: string) => {
    setState((prev) => (prev && prev.id === id ? null : prev));
  }, []);

  const pin = useCallback((id: string) => {
    setState({ id, mode: "pinned" });
  }, []);

  const registerPopup = useCallback((id: string, node: HTMLElement | null) => {
    if (node) popups.current.set(id, node);
    else popups.current.delete(id);
  }, []);

  useEffect(() => {
    if (!state || state.mode !== "pinned") return;

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setState(null);
    };
    const onPointerDown = (e: PointerEvent) => {
      const node = popups.current.get(state.id);
      const target = e.target as Node;
      if (node && node.contains(target)) return;
      const anchor = document.querySelector(`[data-tip-anchor="${state.id}"]`);
      if (anchor && anchor.contains(target)) return;
      setState(null);
    };

    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointerDown, true);
    };
  }, [state]);

  const value: TooltipController = {
    openId: state?.id ?? null,
    openMode: state?.mode ?? null,
    open,
    close,
    pin,
    registerPopup,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
