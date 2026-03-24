const SITE_TEXTURE =
  "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48ZmlsdGVyIGlkPSJuIj48ZmVUdXJidWxlbmNlIHR5cGU9ImZyYWN0YWxOb2lzZSIgYmFzZUZyZXF1ZW5jeT0iMC45IiBudW1PY3RhdmVzPSI0IiBzdGl0Y2hUaWxlcz0ic3RpdGNoIi8+PC9maWx0ZXI+PHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsdGVyPSJ1cmwoI24pIi8+PC9zdmc+";

export default function SiteBackground() {
  return (
    <div className="fixed inset-0 -z-10" aria-hidden="true">
      {/* Base gradient (145deg reflective) */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage:
            "linear-gradient(145deg, rgb(67, 67, 67) 0%, rgb(67, 67, 67) 25%, rgb(255, 255, 255) 30%, rgb(29, 29, 29) 40%, rgb(29, 29, 29) 80%, rgb(67, 67, 67) 100%)",
        }}
      />
      {/* Dark overlay gradient (-135deg) */}
      <div
        className="absolute inset-0 z-10"
        style={{
          backgroundImage:
            "linear-gradient(-135deg, rgb(0, 0, 0) 0%, rgb(0, 0, 0) 10%, rgb(39, 38, 44) 24%, rgb(48, 47, 54) 26%, rgb(39, 38, 44) 28%, rgb(0, 0, 0) 40%, rgb(0, 0, 0) 100%)",
        }}
      />
      {/* Subtle texture overlay */}
      <div
        className="absolute inset-0 z-10 opacity-[0.04]"
        style={{
          backgroundImage: `url("${SITE_TEXTURE}")`,
          backgroundRepeat: "repeat",
        }}
      />
    </div>
  );
}
