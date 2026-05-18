# Vistas.py is a part of the PYTHIA event generator.
# Copyright (C) 2026 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

# Authors: Philip Ilten, Ahmed Youssef, and Jure Zupan.

# The Vistas class provides a Python implementation to visualize PYTHIA
# events using the HSF Phoenix event display
# (https://hepsoftwarefoundation.org/phoenix).


# ==========================================================================
class Vistas:
    """
    Visualization class. The following example visualizes a single event
    assuming a PYTHIA object 'pythia'.

        pythia.next()
        vistas = pythia8.Vistas(pythia)
        vistas.display()

    The following is an overview of the primary methods for interaction.

    data:    return JSON event data for visualization.
    config:  return JSON event configuration for visualization.
    options: return the default visualization options dictionary.
    display: create the JSON data and configuration, upload to the HSF Phoenix
             server, and display the result.
    write:   write the data and configuration JSON to file.

    The visualization options can be set via the 'opts' member dictionary
    of this class. See the 'options' method for details. The following
    example toggles off all particle categories except the hard process.

        vistas.opts["show"] = ["hard process"]

    The following are relevant members that can be modified in advanced usage.

    pythia:   attached PYTHIA instance, used to get events and event info.
    selector: particle selector with a signature of 'selector(event, idx)'.
              Here, 'idx' is the particle index and 'event' the PYTHIA event.
    pdb:      attached PYTHIA 'ParticleData' object.
    sdb:      status code database.
    cdb:      category database.
    """

    # ----------------------------------------------------------------------
    def __init__(
        self,
        pythia,
        selector=lambda event, idx: len(event[idx].daughterList()) != 1
        or event[idx].statusAbs() == 21,
    ):
        """
        Initializer for the visualizer.

        pythia:   PYTHIA instance to attach to this visualizer.
        selector: particle selector with a signature of 'selector(event, idx)'.
                  Here, 'idx' is the particle index and 'event' the PYTHIA
                  event.
        """
        # Allow this class to be used outside the Pythia 8 library.
        global Vec4, Event, costheta
        try:
            Vec4, Event, costheta
        except NameError:
            try:
                from pythia8 import Vec4, Event, costheta
            except:
                from pythia8mc import Vec4, Event, costheta
        # Local instance of Pythia.
        self.pythia = pythia
        # Particle selector.
        self.selector = selector
        # Particle database.
        self.pdb = pythia.particleData
        # Status code database.
        self.sdb = self.statuses(
            self.pythia.settings.word("xmlPath") + "ParticleProperties.xml"
        )
        # Category database.
        self.cdb = {
            "hard process": {"range": (21, 29)},
            "MPI": {"range": (31, 39)},
            "ISR": {"range": (41, 49)},
            "FSR": {"range": (51, 59)},
            "parton prep": {"range": (71, 79)},
            "hadronization": {"range": (81, 89)},
            "decay": {"range": (91, 99)},
            "beam": {"range": (11, 19)},
            "beam remnants": {"range": (61, 69)},
            "color flow": {
                "range": (0, 0),
            },
            "other": {"range": (0, 0)},
        }
        # Set the colors (rainbow, from red to magenta).
        skip, idx = ["hard process", "color flow", "other"], 0
        for key, val in self.cdb.items():
            if key in skip:
                continue
            val["color"] = Vistas.color(1.5 * idx / (len(self.cdb) - len(skip) - 1) - 1)
            idx += 1
        self.cdb["hard process"]["color"] = Vistas.color(1)
        self.cdb["color flow"]["color"] = Vistas.chroma(-0.6, "000000")
        self.cdb["other"]["color"] = Vistas.chroma(-0.2, "000000")
        # Set the options.
        self.opts = self.options()
        # Set the event number.
        self.idx = 0

    # ----------------------------------------------------------------------
    def data(self, event=None):
        """
        Returns the JSON data needed to visualize an event.

        event: optional PYTHIA event, otherwise from internal PYTHIA object.
        """
        # Increment the event number.
        self.idx += 1

        # Set the original lab frame event.
        if event is not None:
            self.lab = event
        else:
            self.lab = self.pythia.event
        # Copy the event for local moifications like boosting.
        self.bst = Event(self.lab)

        # Reset the graph.
        self.vrts = {}  # Dictionary of vertices.
        self.prts = {}  # Dictionary of particles.
        self.hard = []  # List of hard process vertices.
        self.mpis = []  # List of MPI vertices.
        self.part = []  # List of partially added vertices.
        # Color flow map.
        self.cflow = {"color": {}, "anti-color": {}}
        # Camera position.
        self.camera = [0, 0, 0]

        # Boost the event, if requested.
        if self.opts["frame"] in ["hard process", "beam"]:
            code = 12 if self.opts["frame"] == "beam" else 21
            bst = sum(
                (prt.p() for prt in self.bst if prt.statusAbs() == code),
                Vec4(0, 0, 0, 0),
            )
            bst.flip3()
            self.bst.bst(bst)

        # Create the graph.
        for idx in range(self.bst.size() - 1, 0, -1):
            self.addVertex(idx)
        # Return if there is no hard process.
        if len(self.hard) == 0:
            self.log(2, "no hard process found.")
            return
        # Check the number of MPI match.
        if event is None and self.pythia.infoPython().nMPI() != (
            len(self.hard) + len(self.mpis)
        ):
            self.log(
                1,
                f"{len(self.hard) + len(self.mpis)} of "
                + f"{self.pythia.infoPython().nMPI()} scatters found",
            )

        # Determine the min/max observables per category (if needed).
        obs = [
            self.opts[opt]["observable"]
            for opt in ("length", "color")
            if self.opts[opt]["observable"]
        ]
        if len(obs) > 0:
            self.ocats = {
                cat: {o: [float("inf"), float("-inf")] for o in obs}
                for cat in list(self.cdb.keys()) + ["all"]
            }
            for key, prt in self.prts.items():
                for o in obs:
                    val = getattr(self.lab[prt.key].p(), o)()
                    for cat in (prt.status.name, "all"):
                        oLim = self.ocats[cat][o]
                        oLim[0] = min(val, oLim[0])
                        oLim[1] = max(val, oLim[1])

        # Create the JSON dictionary.
        data = {
            "Pythia 8 Event": {
                "Tracks": {name: [] for name in self.cdb},
                "event number": self.idx,
                "run number": self.pythia.settings.mode("Random:seed"),
            }
        }

        # Save the MPI + hard vertices to JSON, step along x-direction.
        stp = Vec4(self.opts["length"]["factor"] * self.opts["mpi"], 0, 0, 0)
        pos = Vec4(0, 0, 0, 0)
        for idx, mpi in enumerate(self.hard + self.mpis):
            self.saveVertex(dct=data["Pythia 8 Event"]["Tracks"], vrt=mpi, pos=pos)
            pos = Vec4(stp)
            pos.rescale3((-1) ** idx * (idx // 2 + 1))

        # Save the remaining particles to JSON.
        for vrt in self.part:
            self.saveVertex(
                dct=data["Pythia 8 Event"]["Tracks"],
                vrt=vrt,
                pos=vrt.pos,
                reverse=False,
            )

        # Save the color flow to JSON.
        self.saveColorFlow(dct=data["Pythia 8 Event"]["Tracks"]["color flow"])

        # Return the data.
        return data

    # ----------------------------------------------------------------------
    def config(self):
        """
        Create the default Phoenix configuration dictionary.
        """
        level = lambda name, node, toggle, children: {
            "name": name,
            "nodeLevel": node,
            "toggleState": toggle,
            "childrenActive": children,
            "configs": [],
            "children": [],
        }
        cfg = {"phoenixMenu": level("Phoenix Menu", 0, True, True)}
        detector = level("Detector", 1, False, False)
        labels = level("Labels", 1, False, False)
        event = level("Event Data", 1, True, True)
        tracks = level("Tracks", 2, True, True)
        # Turn off the detector elements.
        cfg["phoenixMenu"]["children"].extend([detector, labels, event])
        for sub in ("PST", "Beampipe", "Pixel", "Long Strip", "Short Strip"):
            detector["children"].append(level(sub, 2, False, False))
        # Set up the track categories.
        for cat in self.cdb:
            cat = level(cat, 3, cat in self.opts["show"], False)
            cuts = level("Cut Options", 4, False, False)
            # Set up the cuts.
            cuts["configs"].extend(
                [
                    {"type": "label", "label": "Cuts"},
                    {"type": "button", "label": "Reset cuts"},
                ]
            )
            for obs, obsMin, obsMax, obsStep in (
                ("\u03d5", -3.14, 3.14, 0.01),
                ("\u03b7", -4, 4, 0.1),
                ("pT", 0, 50000, 0.1),
            ):
                cuts["configs"].append(
                    {
                        "type": "rangeSlider",
                        "label": obs,
                        "min": obsMin,
                        "max": obsMax,
                        "step": obsStep,
                        "value": obsMin,
                        "highValue": obsMax,
                        "enableMin": False,
                        "enableMax": False,
                    }
                )
            cat["children"].append(cuts)
            tracks["children"].append(cat)
        event["children"].append(tracks)
        # Set cuts and event display.
        cfg["eventDisplay"] = {
            "cameraPosition": [0, max(250, 1.5 * max(self.camera)), 0],
            "cameraTarget": [0, 0, 0],
            "startClipplingAngle": None,
            "openingClippingAngle": None,
        }
        cfg["cuts"] = {}
        return cfg

    # ----------------------------------------------------------------------
    def options(self):
        """
        Creates the default options for visualizing an event. The options
        are a nested dictionary structure as follows.

        "frame":
           <str>
           boost the event into a specific frame.
           - None: no boost is performed.
           - "hard process": boost to the hard process frame.
           - "beam": boost to the beam frame.
        "show":
           [<str>, ...]
           list of categories passed as strings to toggle on for showing in
           the visualization. By default, all categories are on. If toggled
           off, that category can still be shown by switching the toggle.
        "highlight":
           [<str>, ...]
           list of categories to highlight. All other categories are
           grayed out. If an empty list, then no categories are grayed out.
        "mpi":
           <float>
           controls the sub-collision separation. This factor
           is multiplied by the "length:factor" do determine the separation.
        "length", "color":
           <dct>
           dictionaries that control the length and color attributes of the
           particle lines. The keys are similar for both.
           - "scale":
                <str>
                sets the length or color scale for each particle.
                 - "constant": always use the same color or length.
                 - "log": use a logarithmic scale with a variable skew.
           - "observable":
                <str>
                use this observable for calculating the log scale.
                The value can be a string of any 'Vec4' method from PYTHIA,
                e.g., "e", "pT", etc.
           - "skew":
                <float>
                must be larger/smaller than +1/-1. If +1/-1, then the log
                scale is linear. Larger positive/negative skew stretches the
                smaller/larger values more.
           - "group":
                <str>
                sets the group of categories used for calculating the
                logarithmic scale.
                - "all": use all the categories.
                - "cat": just use the category for that particle.
           - "factor": ('length' only)
                <float>
                multiply the scale by this factor.
           - "offset": ('length' only)
                <float>
                add this value onto the scale.
           - "min"/"max": ('color' only)
                <float>
                limits the chroma values (-1 white/low scale, +1
                black/high scale). If 'max' < 'min' than white/black
                corresponds to high/low scales.
        """
        return {
            "frame": "hard process",
            "show": list(self.cdb.keys()),
            "highlight": [],
            "mpi": 1,
            "length": {
                "scale": "constant",
                "observable": "e",
                "skew": 1e5,
                "group": "cat",
                "factor": 80,
                "offset": 1,
            },
            "color": {
                "scale": "log",
                "observable": "e",
                "skew": 10,
                "min": -0.4,
                "max": 0.4,
                "group": "cat",
            },
        }

    # ----------------------------------------------------------------------
    def display(
        self,
        data=True,
        cfg=True,
        sleep=5,
        pad=0.5,
        upload="FzAX60WBp5hZbZX",
        download="689N2msjNEXY6Dr",
        cernbox="https://cernbox.cern.ch/remote.php/dav/public-files/",
        phoenix="https://hepsoftwarefoundation.org/phoenix/trackml",
    ):
        """
        Create the visualization for an event, upload the data to CERNBOX,
        and open the data with the HSF Phoenix website for display.

        data:     JSON event data, if 'True', create data, otherwise
                  use the provided data.
        cfg:      JSON event configuration. If 'True', create
                  the default configuration. If 'False', do not use a
                  configuration. Otherwise, use the provided JSON
                  dictionary.
        sleep:    time in seconds before deleting the uploaded files.
        pad:      pad event data to this size in MB. This can allow
                  the detector geometry to load in time when using
                  the 'trackml' Phoenix instance.
        upload:   hash to the folder where the files can be uploaded.
        donwload: hash to the folder where the files can be downloaded.
        cernbox:  CERNBOX upload/download link.
        phoenix:  PHOENIX display link.
        """
        import webbrowser, time, urllib.parse, urllib.request

        # Create the event data and configuration if needed.
        if data is True:
            data = self.data()
            self.write("pad", data)
        if cfg is True:
            cfg = self.config()

        # This event padding is a hack, but is used to allow the
        # detector geometry to download before the event data so the
        # event configuration is loaded after the geometry.
        if pad is not None:
            data = self.pad(data=data, key="Pythia 8 Event", target=pad)

        # Upload the event data and configuration (if requested).
        if not data:
            return False
        data = Vistas.upload(data, upload, cernbox)
        if cfg:
            cfg = Vistas.upload(cfg, upload, cernbox)

        # Open the file in the browser.
        args = {"file": f"{cernbox}/{download}/{data}", "type": "json"}
        if cfg:
            args.update({"config": f"{cernbox}/{download}/{cfg}"})
        url = f"{phoenix}?{urllib.parse.urlencode(args)}"
        self.log(0, f"opening the URL {url}")
        # Open in pop-up of Colab.
        try:
            from google.colab.output import eval_js

            eval_js(f"window.open({url!r}, '_blank')", ignore_result=True)
        # Otherwise, open in the browser.
        except ImportError:
            webbrowser.open(url, new=2)

        # Delete the files.
        time.sleep(sleep)
        for path in (data, cfg):
            if not path:
                continue
            request = urllib.request.Request(
                f"{cernbox}/{upload}/{path}",
                method="DELETE",
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            urllib.request.urlopen(request, timeout=120)
        return True

    # ----------------------------------------------------------------------
    def write(self, name, data=True, cfg=True, indent=1):
        """
        Write the visualization for an event to JSON files.

        data:   JSON event data, if 'True', create data, otherwise
                use the provided data.
        cfg:    JSON event configuration. If 'True', create
                the default configuration. Otherwise, use the provided
                JSON dictionary.
        indent: level of indentation for the JSON files.
        """
        import json

        # Create the event data and configuration if needed.
        if data is True:
            data = self.data()
        if cfg is True:
            cfg = self.config()

        # Write the files.
        if type(data) is dict:
            with open(f"{name}_data.json", "w") as out:
                json.dump(data, out, indent=indent)
        if type(cfg) is dict:
            with open(f"{name}_cfg.json", "w") as out:
                json.dump(cfg, out, indent=indent)

    # ----------------------------------------------------------------------
    @staticmethod
    def upload(data, upload, cernbox):
        """
        Upload data to CERNBox.

        data:    JSON data for the upload.
        upload:  hash to the folder where the file can be uploaded.
        cernbox: CERNBOX base link for upload/download.
        """
        import json, uuid, urllib.request, urllib.error

        # Get a unique file name.
        name = f"{uuid.uuid4()}.json"
        data = json.dumps(data).encode("utf-8")
        # Create the request.
        request = urllib.request.Request(
            f"{cernbox}/{upload}/{name}",
            data=data,
            method="PUT",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        # Upload.
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = response.getcode()
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"CERNBOX rejected the upload: HTTP {e.code} {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach CERNBOX: {e.reason}") from e
        if not status in [201, 204]:
            raise RuntimeError(f"CERNBOX rejected the upload with status {status}")
        return name

    # ----------------------------------------------------------------------
    class Status:
        """
        Minimal internal class to store a particle status.

        code: absolute value of PYTHIA status code.
        cat:  integer code for the category.
        name: string code for the category.
        cdb:  internal category database used for code range lookups.
        """

        # ------------------------------------------------------------------
        def __init__(self, code, cdb):
            """
            Initialize the status.
            """
            self.code = abs(code)
            self.cat = len(cdb) - 1
            self.name = "other"
            self.cdb = cdb
            self.update(code)

        # ------------------------------------------------------------------
        def update(self, code):
            """
            Update the particle status, keeping preferred status
            code ordering.
            """
            code = abs(code)
            for cat, (name, dct) in enumerate(self.cdb.items()):
                codeMin, codeMax = dct["range"]
                if codeMin <= code <= codeMax and cat < self.cat:
                    self.code = code
                    self.cat = cat
                    self.name = name
                    return

    # ----------------------------------------------------------------------
    class Vertex:
        """
        Minimal internal class to store a vertex.

        key:  vertex key, either two particle indices '(idx1, idx2)' or a
              single particle index.
        pos:  if saved to JSON, position of this vertex.
        moms: set of mother particles of type Particle.
        dtrs: set of daughter particles of type Particle.
        """

        # ------------------------------------------------------------------
        def __init__(self, key, moms, dtrs):
            """
            Initialize the vertex.
            """
            self.key = key
            self.pos = None
            self.moms = set(moms)
            self.dtrs = set(dtrs)

    # ----------------------------------------------------------------------
    class Particle:
        """
        Minimal internal class to store an edge (particle).

        key:    PYTHIA event index for the particle.
        pos:    if saved to JSON, start and end position of this vertex.
        pro:    production vertex.
        end:    end vertex.
        prt:    PYTHIA particle.
        use:    flag if this particle should be used in the JSON.
        status: status, category and code.
        """

        # ------------------------------------------------------------------
        def __init__(self, key, pro, end, prt, use, cdb):
            """
            Initialize the particle.
            """
            self.key = key
            self.pos = None
            self.pro = pro
            self.end = end
            self.prt = prt
            self.use = use
            self.status = Vistas.Status(prt.status(), cdb)

    # ----------------------------------------------------------------------
    def addVertex(self, idx, dtr=None):
        """
        Add a vertex to the graph.

        idx: event index of the daughter PYTHIA particle for this vertex.
        dtr: original daughter particle, when collapsing a vertex structure.
        """
        key = (self.bst[idx].mother1(), self.bst[idx].mother2())
        if dtr is None:
            dtr = self.addParticle(idx)
        # Return if the vertex has been created.
        if key in self.vrts:
            vrt = self.vrts[key]
            if dtr.use:
                dtr.pro = vrt
                vrt.dtrs.add(dtr)
            return vrt
        # Create the vertex otherwise.
        else:
            # Mother indices sorted by closest angle to daughter direction.
            ddir = self.bst[idx].p()
            idxs = sorted(
                (i for i in self.bst[idx].motherList()),
                key=lambda idx: costheta(ddir, self.bst[idx].p()),
                reverse=True,
            )
            # Return an empty vertex if no mother.
            if len(idxs) == 0:
                return None
            # Find the selected mothers.
            moms = [mom for mom in (self.addParticle(i) for i in idxs) if mom.use]
            # Use the closest mother in angle if no selected mother.
            if len(moms) == 0:
                dtr.status.update(self.bst[idxs[0]].status())
                return self.addVertex(idxs[0], dtr=dtr)
            # Use all mothers if a scatter.
            if all(mom.status.code in (21, 31) for mom in moms):
                vrt = self.Vertex(key=key, moms=moms, dtrs=[dtr])
                self.vrts[key] = vrt
                if moms[0].status.code == 21:
                    self.hard.append(vrt)
                elif moms[0].status.code == 31:
                    self.mpis.append(vrt)
                for mom in moms:
                    mom.end = vrt
            # Otherwise, use the closest selected mother in angle.
            else:
                key = moms[0].key
                if key in self.vrts:
                    vrt = self.vrts[key]
                else:
                    vrt = self.Vertex(key=key, moms=[moms[0]], dtrs=[dtr])
                    self.vrts[key] = vrt
                    moms[0].end = vrt
        # Set the daughter info.
        if dtr.use:
            dtr.pro = vrt
            vrt.dtrs.add(dtr)
        # Return the vertex.
        return vrt

    # ----------------------------------------------------------------------
    def addParticle(self, idx):
        """
        Add a particle to the graph.

        idx: event index for the PYTHIA particle to add.
        """
        key = idx
        # Return if the particle has been created.
        if key in self.prts:
            prt = self.prts[key]
        # Create the particle otherwise.
        else:
            prt = self.Particle(
                key=key,
                pro=None,
                end=None,
                prt=self.bst[idx],
                use=self.selector is None or self.selector(self.bst, idx),
                cdb=self.cdb,
            )
            self.prts[key] = prt
        # Return the particle.
        return prt

    # ----------------------------------------------------------------------
    def saveVertex(self, dct, vrt, pos, reverse=True):
        """
        Save the particle tree from this vertex to a JSON dictionary.

        dct:     JSON dictionary to save the particles.
        vrt:     Vista vertex with particle tree to save.
        pos:     starting position of the particle tree.
        reverse: if 'True' also trace the mother particle line.
        """
        # Add the outgoing particle tree.
        origin = Vec4(0, 0, 0, 0)
        for dtr in vrt.dtrs:
            self.saveParticle(dct=dct, prt=dtr, pos=pos)
        # Add the incoming particle tree if requested.
        if reverse:
            for mom in vrt.moms:
                self.saveParticle(dct=dct, prt=mom, pos=pos, reverse=True)

    # ----------------------------------------------------------------------
    def saveParticle(self, dct, prt, pos, reverse=False):
        """
        Save the particle and its tree to a JSON dictionary.

        dct:     JSON dictionary to save the particles.
        prt:     Vista particle with particle tree to save.
        pos:     starting position of the particle tree.
        reverse: if 'True' store the vertices that are partially saved
                 when tracing the mother particle line.
        """
        # Return if the particle should not be used.
        if not prt.use or prt.pos is not None:
            return

        # Set the step size.
        att = self.opts["length"]
        nrm = att["factor"] * self.strength(prt, att) + att["offset"]
        stp = prt.prt.p()
        if stp.pAbs() == 0:
            stp = Vec4(0, 0.5 * nrm, 0, 0)
        else:
            stp.rescale3(0.5 * nrm / stp.pAbs())
        if reverse:
            stp.flip3()
        # Set the mid and end point of the vector.
        mid = pos + stp
        end = mid + stp
        # Update the camera view.
        for idx in [0, 1, 2]:
            self.camera[idx] = max(abs(end[idx + 1]), self.camera[idx])

        # Perform highlighting if requested.
        highlight = self.opts["highlight"]
        if len(highlight) > 0 and prt.status.name not in highlight:
            color = self.cdb["other"]["color"]
        # Otherwise, set the line color.
        else:
            att = self.opts["color"]
            dif = att["max"] - att["min"]
            nrm = dif * self.strength(prt, att) + att["min"]
            color = Vistas.chroma(nrm, self.cdb[prt.status.name]["color"])

        # Set the particle data.
        cc, ac = prt.prt.col(), prt.prt.acol()
        p = self.lab[prt.key].p()
        pid = prt.prt.id()
        dct[prt.status.name].append(
            {
                "category": prt.status.name,
                "name (PDG ID)": f"{self.pdb.name(pid)} ({pid})",
                "status": f"{self.sdb[prt.status.code]} ({prt.status.code})",
                "px": p.px(),
                "py": p.py(),
                "pz": p.pz(),
                "E": p.e(),
                "pT": p.pT(),
                "eta": p.eta(),
                "phi": p.phi(),
                "charge x 3": prt.prt.chargeType(),
                "color index": (cc, ac),
                "index": prt.key,
                "pos": [
                    [pos[i] for i in (1, 2, 3)],
                    [mid[i] for i in (1, 2, 3)],
                    [end[i] for i in (1, 2, 3)],
                ],
                "color": color,
            }
        )

        # Save the position of the particle.
        prt.pos = (pos, end)
        # Save the color flow of the particle if endpoint.
        if cc != 0 or ac != 0:
            # Only color endpoint if daughters are all colorless.
            if (
                prt.end is None
                or sum(abs(dtr.prt.colType()) for dtr in prt.end.dtrs) == 0
            ):
                if cc != 0:
                    self.cflow["color"][cc] = prt
                if ac != 0:
                    self.cflow["anti-color"][ac] = prt

        # Add the particle tree.
        vrt = prt.pro if reverse else prt.end
        if vrt is not None:
            dtrs = vrt.moms if reverse else vrt.dtrs
            for dtr in dtrs:
                self.saveParticle(dct, dtr, end, reverse)
            # Store the forward tree when in reverse for adding later.
            if reverse:
                for dtr in vrt.dtrs:
                    if dtr.use and dtr.pos is None:
                        vrt.pos = Vec4(pos)
                        self.part.append(vrt)

    # ----------------------------------------------------------------------
    def saveColorFlow(self, dct):
        """
        Save the color flow lines.

        dct: JSON dictionary to save the color flow lines.
        """
        # Determine the line color.
        color = self.cdb["color flow"]["color"]
        if len(self.opts["highlight"]) > 0:
            if "color flow" not in self.opts["highlight"]:
                color = self.cdb["other"]["color"]

        # Loop over the color lines.
        for cc, pos in self.cflow["color"].items():
            # Set the positions.
            if cc in self.cflow["anti-color"]:
                pos = pos.pos[1]
                end = self.cflow["anti-color"][cc].pos[1]
                bow = pos + end
                mid = end - pos
                bow.rescale3(0.5 * mid.pAbs() / bow.pAbs())
                mid.rescale3(0.5)
                mid = mid + pos + bow
            else:
                self.log(1, f"missing anti-color {cc}.")
                continue
            # Add the line.
            dct.append(
                {
                    "category": "color flow",
                    "color index": cc,
                    "pos": [
                        [pos[i] for i in (1, 2, 3)],
                        [mid[i] for i in (1, 2, 3)],
                        [end[i] for i in (1, 2, 3)],
                    ],
                    "color": color,
                }
            )

        # Check the anti-color.
        for ac in self.cflow["anti-color"]:
            if not ac in self.cflow["color"]:
                self.log(1, f"missing color {ac}.")

    # ----------------------------------------------------------------------
    def strength(self, prt, att):
        """
        Calculate the strength of a particle, from 0 to 1, for a given
        observable.

        prt: Vista particle to calcualte the strength for.
        att: physical attribute dictionary, for either 'length' or 'color'.
        """
        import math

        scale = att["scale"]
        cat = "all" if att["group"] == "all" else prt.status.name
        if scale == "constant":
            return 1
        o = att["observable"]
        oMin, oMax = self.ocats[cat][o]
        oPrt = getattr(self.lab[prt.key].p(), o)()
        t = (oPrt - oMin) / (oMax - oMin) if oMin != oMax else 1
        k = att["skew"]
        if k > 0:
            k = max(1.01, k)
            return math.log1p((k - 1) * t) / math.log(k)
        else:
            k = abs(min(-1.01, k))
            return 1 - math.log1p((k - 1) * (1 - t)) / math.log(k)

    # ----------------------------------------------------------------------
    def log(self, level, message):
        """
        Simple utility to print a log message.
        """
        levels = {0: "INFO", 1: "WARNING", 2: "ERROR"}
        if level in levels:
            print(f"VISTAS {levels[level]}: {message}"),

    # ----------------------------------------------------------------------
    @staticmethod
    def chroma(t, color):
        """
        Lighten or darken a hex color.

        t:     scale to lighten/darken between -1 (to white) and 1 (to black).
        color: hex color to lighten or darken.
        """
        import colorsys

        r, g, b = (int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))
        hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
        ll = max(0.0, min(1.0, ll - t))
        r, g, b = colorsys.hls_to_rgb(hh, ll, ss)
        return "{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

    # ----------------------------------------------------------------------
    @staticmethod
    def color(t):
        """
        Return a hex color along the rainbow spectrum.

        t: scale to between -1 (red) and 1 (magenta).
        """
        import colorsys

        f = (max(-1.0, min(1.0, t)) + 1) / 2
        r, g, b = colorsys.hls_to_rgb(f * 5 / 6, 0.5, 1.0)
        return "{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

    # ----------------------------------------------------------------------
    @staticmethod
    def statuses(path):
        """
        Return a dictionary of descriptions for PYTHIA status codes.
        The key is the absolute value of the status code and the value is
        the text description.

        path: path to the file 'ParticleProperties.xml' which contains
              the status codes in XML format.
        """
        import re, html

        # Simple function to clean the code.
        clean = lambda s: re.sub(
            r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s))
        ).strip()

        # Load the XML and get the status code block.
        with open(path, encoding="utf-8") as xml:
            raw = xml.read()
        block = re.search(
            r'<method name="int Particle::status\(\)">(.*?)</method>', raw, re.DOTALL
        ).group(1)

        # Regular expression for list items.
        reLi = re.compile(r"<li>\s*(.*?)\s*(?:</li>|<ul>)", re.DOTALL)
        # Regular expression for status code groups.
        reGroup = re.compile(r"^\s*\d+\s*-\s*\d*\s*:\s*(.*)$", re.DOTALL)
        # Regular expression for status codes.
        reCodes = re.compile(r"^\s*(\d+(?:\s*,\s*\d+)*)\s*:\s*(.*)$", re.DOTALL)

        # Loop over the list items.
        sdb, group = {}, None
        for li in reLi.findall(block):
            # Check if a group match.
            matchGroup = reGroup.match(li)
            if matchGroup:
                group = clean(matchGroup.group(1))
            # Check if a status code match.
            else:
                matchCodes = reCodes.match(li)
                if matchCodes:
                    txt = clean(matchCodes.group(2))
                    # Loop over the status codes.
                    for code in re.split(r",\s*", matchCodes.group(1)):
                        sdb[int(code)] = "[" + group + "] " + txt
        return sdb

    # ----------------------------------------------------------------------
    @staticmethod
    def pad(data, key, target):
        """
        Pad a JSON data structure until it reaches a specified size.

        data:   JSON data.
        key:    data entry to duplicate until the target size is reached.
        target: target size in MB.
        """
        import json

        size = len(json.dumps(data[key], ensure_ascii=False).encode("utf-8"))
        target *= 1024**2

        # Each entry contributes '"<key>": <value>' and ', ' between entries.
        now, n = 2, 0
        while now < target:
            n += 1
            new = f"{key} {n}"
            now += len(new.encode("utf-8")) + 4 + size + (2 if n > 1 else 0)
            data[new] = data[key]
        return data
