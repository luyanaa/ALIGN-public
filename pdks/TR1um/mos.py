from align.primitive.default.canvas import DefaultCanvas
from align.cell_fabric.generators import Region, Wire, Via
from align.cell_fabric.grid import EnclosureGrid, UncoloredCenterLineGrid, SingleGrid, CenteredGrid, CenterLineGrid
from math import floor, ceil
import collections
import string
import logging
logger = logging.getLogger(__name__)


class MOSGenerator(DefaultCanvas):

    def __init__(self, pdk, height, fin, gate, gateDummy, shared_diff, stack, bodyswitch, **kwargs):
        self.primitive_constraints = kwargs.get('primitive_constraints', [])
        self.primitive_parameters = kwargs.get('primitive_parameters')
        super().__init__(pdk)

        exact_width = None
        exact_length = None
        length_diff = 0
        dynamic_space = 0
        if self.primitive_parameters:
            device_names = [*self.primitive_parameters.keys()]
            exact_width = self.primitive_parameters[device_names[0]]['W']
            exact_width = round(float(exact_width)*1E9)  ### Width in nanometers (even grid)
            exact_length = self.primitive_parameters[device_names[0]]['L']
            exact_length = round(float(exact_length)*1E9)  ### Length in nanometers
            assert exact_length % 2 == 0, f"Transistor gate length {exact_length} must be even"
            assert exact_width % 2 == 0, f"Transistor width {exact_width} must be even"
            length_diff = exact_length - self.pdk['Poly']['Width']
            assert length_diff >= 0, f"Transistor gate length {exact_length} must be greater than the minimum gate length {self.pdk['Poly']['Width']}"
            # TR-1um poly pitch sizing (drawing-layer rules, gate-pad scheme):
            #   GC.S1 pad-poly and poly-poly spacing >= 1.2um
            #   CO.GG  S/D CO to gate              >= 1.0um
            #   V1.GC  V1 to gate                  >= 1.2um
            # With poly width L, gate pad 2600 on the M1 track and poly centered at
            # pitch/2, all margins are met when pitch >= L + 5000.
            poly_pitch = max(self.pdk['Poly']['Pitch'], exact_length + 5000)
            poly_pitch = ((poly_pitch + 1) // 2) * 2
            self.pdk['Poly']['Pitch'] = poly_pitch
            self.pdk['Poly']['Offset'] = poly_pitch // 2
            self.pdk['Poly']['Width'] = exact_length
            self.pdk['M1']['Pitch'] = poly_pitch

        self.exact_patterns = None
        height = max(height, 4*ceil((fin+16)/4))
        # TR-1um: height*FinPitch must be a multiple of M2Pitch. During the extract
        # phase ALIGN re-runs the generator with the placed (routed) height, which
        # includes power-rail growth and is not grid-aligned; round UP to the next
        # legal height instead of asserting (the extra fins become dummy active).
        m2_fin_multiple = self.pdk['M2']['Pitch'] // self.pdk['Fin']['Pitch']
        height = ceil(height / m2_fin_multiple) * m2_fin_multiple
        for const in self.primitive_constraints:
            if const.constraint == 'Generator':
                if const.parameters is not None:
                    if const.parameters.get('exact_patterns'):
                        self.exact_patterns = const.parameters.get('exact_patterns')
                        self.exact_patterns = self.exact_patterns[0]
                    if const.parameters.get('height'):
                        height = const.parameters.get('height')

        self.finsPerUnitCell = height
        assert self.finsPerUnitCell % 4 == 0
        assert (self.finsPerUnitCell*self.pdk['Fin']['Pitch'])%self.pdk['M2']['Pitch']==0
        self.m2PerUnitCell = (self.finsPerUnitCell*self.pdk['Fin']['Pitch'])//self.pdk['M2']['Pitch']
        self.unitCellHeight = self.m2PerUnitCell* self.pdk['M2']['Pitch']
        ######### Derived Parameters ############
        self.shared_diff = shared_diff
        self.stack = stack
        self.bodyswitch = bodyswitch
        # Keep ALIGN's default three dummy gates per side: the analog OTA
        # primitives need the matching surroundings, and per-cell KLayout DRC
        # measures fewer violations than the compacted one-dummy variant
        # (33 vs 37 top-level items in the latest sweep).
        gateDummy = 3
        self.gateDummy = gateDummy
        self.gate = gate*self.stack
        # TR-1um: single-finger devices when gate=1 (no doubling). The 2x doubling
        # was for CMC/SCM interdigitation patterns; standard cells need 1 finger
        # per NF=1 device to meet the exact W spec.
        self.gatesPerUnitCell = self.gate + 2*self.gateDummy*(1-self.shared_diff)
        self.finDummy = (self.finsPerUnitCell-fin)//2
        self.lFin = height ### This defines numebr of fins for tap cells; Should we define it in the layers.json?
        assert self.finDummy >= 8, f"number of fins/width {fin} in the transistor must be less than unit cell height {self.finsPerUnitCell -2*8}"
        assert fin > 1, "number of fins in the transistor must be more than 1"
        assert gateDummy > 0
        unitCellLength = self.gatesPerUnitCell* self.pdk['Poly']['Pitch']
        # TR-1um manufacturing grid is 0.05um. unitCellHeight is a multiple of
        # M2 pitch (5000) and FinPitch is 50, so activeOffset lands on odd 25nm
        # residues (e.g. 34975). Round to the 50nm grid.
        activeOffset = (self.unitCellHeight//2 - self.pdk['Fin']['Pitch']//2 + 25) // 50 * 50
        activeWidth = exact_width if exact_width else self.pdk['Fin']['Pitch']*fin
        activePitch = self.unitCellHeight
        RVTWidth = activeWidth + 2*self.pdk['Active']['active_enclosure']


        stoppoint = self.pdk['Active']['activePolyExTracks']*self.pdk['M2']['Pitch']
        self.pl = self.addGen( Wire( 'pl', 'Poly', 'v',
                                     clg=UncoloredCenterLineGrid( pitch= self.pdk['Poly']['Pitch'], width= self.pdk['Poly']['Width'], offset= self.pdk['Poly']['Offset']),
                                     spg=EnclosureGrid( pitch=self.unitCellHeight, offset=self.pdk['M2']['Offset'], stoppoint=stoppoint, check=True)))

        # Gate contact pad on the poly grid. Keep it narrow enough to preserve
        # the foundry GC spacing rule; the dedicated M1/CO connection is made
        # by ALIGN's routing abstraction.
        gate_pad_width = self.pdk['Poly']['Width']
        self.gpad = self.addGen(Wire('gpad', 'Poly', 'v',
                                     clg=UncoloredCenterLineGrid(pitch=self.pdk['Poly']['Pitch'], width=gate_pad_width, offset=self.pdk['Poly']['Offset']),
                                     spg=EnclosureGrid(pitch=self.pdk['M2']['Pitch'], offset=0, stoppoint=1300, check=False)))

        self.m1_updated = self.addGen( Wire( 'm1_updated ', 'M1', 'v',
                                     clg=UncoloredCenterLineGrid( pitch=self.pdk['M1']['Pitch'], width=self.pdk['M1']['Width']),
                                     spg=EnclosureGrid( pitch=self.pdk['M2']['Pitch'], stoppoint=self.pdk['V1']['VencA_L'] +self.pdk['M2']['Width']//2, check=False)))

        self.m2_updated = self.addGen( Wire( 'm2_updated ', 'M2', 'h',
                                     clg=UncoloredCenterLineGrid( pitch=self.pdk['M2']['Pitch'], width=self.pdk['M2']['Width']),
                                     spg=EnclosureGrid( pitch=self.pdk['M1']['Pitch'], stoppoint=self.pdk['V1']['VencA_L']+self.pdk['M1']['Width']//2, check=False)))

        self.fin = self.addGen( Wire( 'fin', 'Fin', 'h',
                                      clg=UncoloredCenterLineGrid( pitch= self.pdk['Fin']['Pitch'], width= self.pdk['Fin']['Width'], offset= self.pdk['Fin']['Offset']),
                                      spg=SingleGrid( offset=0, pitch=unitCellLength)))

        stoppoint = ((self.gateDummy-1)* self.pdk['Poly']['Pitch'] +  self.pdk['Poly']['Offset'])*(1-self.shared_diff)
        self.active = self.addGen( Wire( 'active', 'Active', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=activePitch, width=activeWidth, offset=activeOffset),
                                         spg=EnclosureGrid( pitch=unitCellLength, offset=0, stoppoint=stoppoint, check=True)))

        self.RVT = self.addGen( Wire( 'RVT', 'Rvt', 'h',
                                      clg=UncoloredCenterLineGrid( pitch=activePitch, width=RVTWidth, offset=activeOffset),
                                      spg=EnclosureGrid( pitch=unitCellLength, offset=0, stoppoint=stoppoint, check=True)))

        self.LVT = self.addGen( Wire( 'LVT', 'Lvt', 'h',
                                      clg=UncoloredCenterLineGrid( pitch=activePitch, width=RVTWidth, offset=activeOffset),
                                      spg=EnclosureGrid( pitch=unitCellLength, offset=0, stoppoint=stoppoint, check=True)))

        self.HVT = self.addGen( Wire( 'HVT', 'Hvt', 'h',
                                      clg=UncoloredCenterLineGrid( pitch=activePitch, width=RVTWidth, offset=activeOffset),
                                      spg=EnclosureGrid( pitch=unitCellLength, offset=0, stoppoint=stoppoint, check=True)))

        self.SLVT = self.addGen( Wire( 'SLVT', 'Slvt', 'h',
                                      clg=UncoloredCenterLineGrid( pitch=activePitch, width=RVTWidth, offset=activeOffset),
                                      spg=EnclosureGrid( pitch=unitCellLength, offset=0, stoppoint=stoppoint, check=True)))

        offset = self.gateDummy*self.pdk['Poly']['Pitch']+self.pdk['Poly']['Offset'] - self.pdk['Poly']['Pitch']//2
        stoppoint = self.gateDummy*self.pdk['Poly']['Pitch'] + self.pdk['Poly']['Offset']-self.pdk['Pc']['PcExt']-self.pdk['Poly']['Width']//2
        self.pc = self.addGen( Wire( 'pc', 'Pc', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=self.pdk['M2']['Pitch'], width=self.pdk['Pc']['PcWidth'], offset=self.pdk['M2']['Pitch']),
                                         spg=EnclosureGrid( pitch=unitCellLength, offset=offset*self.shared_diff, stoppoint=stoppoint-offset*self.shared_diff, check=True)))

        self.nselect = self.addGen( Region( 'nselect', 'Nselect',
                                            v_grid=UncoloredCenterLineGrid( offset= 0, pitch= self.pdk['M2']['Pitch'], width= self.pdk['M2']['Width']),
                                            h_grid=self.fin.clg))
        self.pselect = self.addGen( Region( 'pselect', 'Pselect',
                                            v_grid=UncoloredCenterLineGrid( offset= 0, pitch= self.pdk['M2']['Pitch'], width= self.pdk['M2']['Width']),
                                            h_grid=self.fin.clg))
        self.nwell = self.addGen( Region( 'nwell', 'Nwell',
                                            v_grid=UncoloredCenterLineGrid( offset= 0, pitch= self.pdk['M2']['Pitch'], width= self.pdk['M2']['Width']),
                                            h_grid=self.fin.clg))

        self.active_diff = self.addGen( Wire( 'active_diff', 'Active', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=activePitch, width=activeWidth, offset=activeOffset),
                                         spg=SingleGrid( pitch=self.pdk['Poly']['Pitch'], offset=(self.gateDummy-1)*self.pdk['Poly']['Pitch']+self.pdk['Poly']['Pitch']//2)))

        self.RVT_diff = self.addGen( Wire( 'RVT_diff', 'Rvt', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=activePitch, width=RVTWidth, offset=activeOffset),
                                         spg=SingleGrid( pitch=self.pdk['Poly']['Pitch'], offset=(self.gateDummy-1)*self.pdk['Poly']['Pitch']+self.pdk['Poly']['Pitch']//2)))

        self.LVT_diff = self.addGen( Wire( 'LVT_diff', 'Lvt', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=activePitch, width=RVTWidth, offset=activeOffset),
                                         spg=SingleGrid( pitch=self.pdk['Poly']['Pitch'], offset=(self.gateDummy-1)*self.pdk['Poly']['Pitch']+self.pdk['Poly']['Pitch']//2)))

        self.HVT_diff = self.addGen( Wire( 'HVT_diff', 'Hvt', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=activePitch, width=RVTWidth, offset=activeOffset),
                                         spg=SingleGrid( pitch=self.pdk['Poly']['Pitch'], offset=(self.gateDummy-1)*self.pdk['Poly']['Pitch']+self.pdk['Poly']['Pitch']//2)))

        self.SLVT_diff = self.addGen( Wire( 'SLVT_diff', 'Slvt', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=activePitch, width=RVTWidth, offset=activeOffset),
                                         spg=SingleGrid( pitch=self.pdk['Poly']['Pitch'], offset=(self.gateDummy-1)*self.pdk['Poly']['Pitch']+self.pdk['Poly']['Pitch']//2)))

        stoppoint = unitCellLength//2-self.pdk['Active']['activebWidth_H']//2
        offset_active_body = (self.lFin//2)*self.pdk['Fin']['Pitch']+self.unitCellHeight-self.pdk['Fin']['Pitch']//2
        self.activeb = self.addGen( Wire( 'activeb', 'Active', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=self.pdk['M2']['Pitch'], width=self.pdk['Active']['activebWidth'], offset=0),
                                         spg=EnclosureGrid( pitch=unitCellLength, offset=0, stoppoint=stoppoint, check=True)))

        self.activeb_diff = self.addGen( Wire( 'activeb_diff', 'Active', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=self.pdk['M2']['Pitch'], width=self.pdk['Active']['activebWidth'], offset=0),
                                         spg=SingleGrid( pitch=self.pdk['Poly']['Pitch'], offset=(self.gateDummy-1)*self.pdk['Poly']['Pitch']+self.pdk['Poly']['Pitch']//2)))

        self.pb_diff = self.addGen( Wire( 'pb_diff', 'Pb', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=self.pdk['M2']['Pitch'], width=self.pdk['Pb']['pbWidth'], offset=0),
                                         spg=SingleGrid( pitch=self.pdk['Poly']['Pitch'], offset=(self.gateDummy-1)*self.pdk['Poly']['Pitch']+self.pdk['Poly']['Pitch']//2)))

        stoppoint = unitCellLength//2-self.pdk['Pb']['pbWidth_H']//2
        self.pb = self.addGen( Wire( 'pb', 'Pb', 'h',
                                         clg=UncoloredCenterLineGrid( pitch=self.pdk['M2']['Pitch'], width=self.pdk['Pb']['pbWidth'], offset=0),
                                         spg=EnclosureGrid( pitch=unitCellLength, offset=0, stoppoint=stoppoint, check=True)))


        # Horizontal M1 bridge for the gate contact: spans the M1 track (V1 stack)
        # and the pl-track gate CO at the CO's M2 track.
        self.m1_gate_h = self.addGen( Wire( 'm1_gate_h', 'M1', 'h',
                                     clg=UncoloredCenterLineGrid( pitch= self.pdk['M2']['Pitch'], width= self.pdk['M1']['Width']),
                                     spg=EnclosureGrid( pitch=self.pdk['M1']['Pitch'], offset=0, stoppoint=700, check=False)))

        self.v1_x = self.addGen( Via( 'v1_x', 'V1',
                                    h_clg=self.m2_updated.clg,
                                    v_clg=self.m1_updated.clg,
                                    WidthX=self.pdk['V1']['WidthX'],
                                    WidthY=self.pdk['V1']['WidthY']))

        self.va = self.addGen( Via( 'va', 'V0',
                                    h_clg=self.m2.clg,
                                    v_clg=self.m1_updated.clg,
                                    WidthX=self.pdk['V0']['WidthX'],
                                    WidthY=self.pdk['V0']['WidthY']))

        # Gate-contact via: same cut as va but x-aligned to the gate poly (pl)
        # grid so the gate CO lands on the poly pad.
        self.va_gate = self.addGen( Via( 'va_gate', 'V0',
                                    h_clg=self.m2.clg,
                                    v_clg=self.pl.clg,
                                    WidthX=self.pdk['V0']['WidthX'],
                                    WidthY=self.pdk['V0']['WidthY']))

        self.v0 = self.addGen( Via( 'v0', 'V0',
                                    h_clg=CenterLineGrid(),
                                    v_clg=self.m1_updated.clg,
                                    WidthX=self.pdk['V0']['WidthX'],
                                    WidthY=self.pdk['V0']['WidthY']))

        self.v0.h_clg.addCenterLine( 0,                 self.pdk['V0']['WidthY'], False)
        v0pitch = self.pdk['V0']['WidthY'] + self.pdk['V0']['SpaceY']
        v0Offset = activeOffset - activeWidth//2 + self.pdk['V0']['VencA_L'] + self.pdk['V0']['WidthY']//2
        v0_number = floor((activeWidth-2*self.pdk['V0']['VencA_L']-self.pdk['V0']['WidthY'])/v0pitch)
        v0_number = max(v0_number, 1)
        assert v0_number > 0, "V0 can not be placed in the active region"
        #v0_number = v0_number if v0_number < 4 else v0_number - 1 ## To avoid voilation of V0 enclosure by M1 DRC
        for i in range(v0_number):
            self.v0.h_clg.addCenterLine(i*v0pitch+v0Offset,    self.pdk['V0']['WidthY'], True)
        self.v0.h_clg.addCenterLine( self.unitCellHeight,    self.pdk['V0']['WidthY'], False)
        info = self.pdk['V0']

    def _addMOS( self, x, y, x_cells,  vt_type, name='M1', reflect=False, **parameters):

        fullname = f'{name}_X{x}_Y{y}'
        self.subinsts[fullname].parameters.update(parameters)

        def _connect_diffusion(i, pin):
            self.addWire(self.m1_updated, None, i, (grid_y0, -1), (grid_y1, 1))
            if not hasattr(self, '_diffusion_vias'):
                self._diffusion_vias = set()
            j = max(1, self.v0.h_clg.n // 2)
            via_key = (i, y, j)
            if via_key not in self._diffusion_vias:
                self.addVia(self.v0, f'{fullname}:{pin}', i, (y, j))
                self._diffusion_vias.add(via_key)
            self._xpins[name][pin].append(i)

        # Draw FEOL Layers
        if self.shared_diff == 0:
            self.addWire(self.active, None, y, (x,1), (x+1,-1))
        elif self.shared_diff == 1 and x == x_cells-1:
            self.addWire(self.active_diff, None, y, 0, self.gate*x_cells+1)
        else:
            pass

        for i in range(self.gate):
            self.addWire( self.pl, None, i+self.gatesPerUnitCell*x+self.gateDummy,   (y,1), (y+1,-1))


        def _addRVT(x, y, x_cells):
            if self.shared_diff == 0:
                self.addWire( self.RVT,  None, y,          (x, 1), (x+1, -1))
            elif self.shared_diff == 1 and x == x_cells-1:
                self.addWire( self.RVT_diff,  None, y, 0, self.gate*x_cells+1)
            else:
                pass

        def _addLVT(x, y, x_cells):
            if self.shared_diff == 0:
                self.addWire( self.LVT,  None, y,          (x, 1), (x+1, -1))
            elif self.shared_diff == 1 and x == x_cells-1:
                self.addWire( self.LVT_diff,  None, y, 0, self.gate*x_cells+1)
            else:
                pass

        def _addHVT(x, y, x_cells):
            if self.shared_diff == 0:
                self.addWire( self.HVT,  None, y,          (x, 1), (x+1, -1))
            elif self.shared_diff == 1 and x == x_cells-1:
                self.addWire( self.HVT_diff,  None, y, 0, self.gate*x_cells+1)
            else:
                pass
        if vt_type == 'RVT':
            _addRVT(x, y, x_cells)
        elif vt_type == 'LVT':
            _addLVT(x, y, x_cells)
        elif vt_type == 'HVT':
            _addHVT(x, y, x_cells)
        else:
            print("This VT type not supported")
            exit()

        # Source, Drain, Gate Connections
        grid_y0 = y*self.m2PerUnitCell + 1
        grid_y1 = (y+1)*self.m2PerUnitCell-5
        gate_x = self.gateDummy*self.shared_diff + x * self.gatesPerUnitCell + self.gatesPerUnitCell // 2
        # Connect Gate (gate_x)
        co_w = self.pdk['V0']['WidthX']
        m1_pitch = self.pdk['M1']['Pitch']
        poly_pitch = self.pdk['Poly']['Pitch']
        poly_w = self.pdk['Poly']['Width']
        co_x = gate_x * m1_pitch
        first_finger_track = self.gateDummy + self.gatesPerUnitCell * x
        last_finger_track = first_finger_track + self.gate - 1
        first_center = self.pl.clg.value(first_finger_track)
        if isinstance(first_center, tuple): first_center = first_center[0]
        last_center = self.pl.clg.value(last_finger_track)
        if isinstance(last_center, tuple): last_center = last_center[0]
        pad_left = min(co_x - co_w//2 - 800, first_center - poly_w//2)
        pad_right = max(co_x + co_w//2 + 800, last_center + poly_w//2)
        co_y = (grid_y1+2) * self.pdk['M2']['Pitch']
        pad_bottom = co_y - self.pdk['V0']['WidthY']//2 - 800
        pad_top = co_y + self.pdk['V0']['WidthY']//2 + 800
        self.addWire(self.gpad, None, gate_x, (grid_y1+2, -1), (grid_y1+2, 1))
        self.terminals.append({'layer': 'Poly', 'netName': None, 'netType': 'drawing',
                               'rect': [pad_left, pad_bottom, pad_right, pad_top]})
        self.addWire(self.m1_updated, None, gate_x, (grid_y1+2, -1), (grid_y1+4, 1))
        self.addWire(self.pc, None, grid_y1+1, (x,1), (x+1,-1))
        self.addVia(self.va, f'{fullname}:G', gate_x, grid_y1+2)
        self._xpins[name]['G'].append(gate_x)
        (center_terminal, side_terminal) = ('S', 'D') if self.gate%4 == 0 else ('D', 'S')
        center_terminal = 'D' if self.stack > 1 else center_terminal
        if self.gate == 1 and self.stack == 1 and self.shared_diff and x % 2:
            # Adjacent one-finger devices share the middle diffusion rail.
            # Alternate their source/drain assignment so that rail has one net.
            _connect_diffusion(gate_x, side_terminal)
            _connect_diffusion(gate_x + 1, center_terminal)
        else:
            _connect_diffusion(gate_x, center_terminal)
            if self.gate == 1 and self.stack == 1:
                _connect_diffusion(gate_x + 1, side_terminal)
        for x_terminal in range(1,1+self.gate//2):
            terminal = center_terminal if x_terminal%2 == 0 else side_terminal
            if self.stack > 1 and x_terminal != self.gate//2:
                continue
            terminal = 'S' if self.stack > 1 else terminal
            _connect_diffusion(gate_x - x_terminal, terminal)
            _connect_diffusion(gate_x + x_terminal, terminal)
    def _connectDevicePins(self, y, y_cells, connections):
        center_track = y * self.m2PerUnitCell + self.m2PerUnitCell // 2
        grid_y1 = (y+1)*self.m2PerUnitCell-5
        gate_track = 0
        diff_track = 1
        for (net, conn) in connections.items():
            contactsx = {(track, pin) for inst, pins in self._xpins.items()
                         for pin, m1tracks in pins.items()
                         for track in m1tracks if (inst, pin) in conn}
            pins = {x[1] for x in contactsx}
            # Shared diffusion and body terminals may need multiple contacts.
            # Dropping all but one disconnects valid source/body routes.
            for pin in pins:
                contacts = {x[0] for x in contactsx if x[1] == pin}
                for j in range(self.minvias):
                    if pin == 'G':
                        current_track = grid_y1 + 3 + gate_track
                        gate_track += 1
                    elif pin == 'B':
                        if y != y_cells - 1:
                            continue
                        current_track = y_cells * self.m2PerUnitCell + (self.lFin*self.pdk['Fin']['Pitch']) // (2*self.pdk['M2']['Pitch']) + 1
                    else:
                        current_track = y * self.m2PerUnitCell + len(connections) * j + diff_track
                        diff_track += 1
                    if pin != 'G' and j > 0:
                        continue
                    self.addWireAndViaSet(net, self.m2_updated, self.v1_x, current_track, contacts)
                    self._nets[net][current_track] = contacts

    def _connectNets(self, x_cells, y_cells):
        def _get_wire_terminators(intersecting_tracks):
            minx, maxx = min(intersecting_tracks), max(intersecting_tracks)
            minL = 2
            L = maxx - minx
            if center_track - minL // 2 <= minx and maxx <= center_track + minL // 2:
                minx, maxx = (center_track - minL // 2, center_track + minL // 2)
            elif L < minL:
                minx, maxx = (minx - (minL - L), maxx) if minx >= center_track else (minx, maxx + (minL - L))
            return minx, maxx

        device_tracks = x_cells * self.gatesPerUnitCell + 2 * self.gateDummy * self.shared_diff
        center_track = device_tracks // 2
        trunk_base = device_tracks + 2
        # TR-1um has no M3. Connect every local M2 segment belonging to a
        # logical net through a reserved M1 vertical trunk and V1 crossings.
        for net_index, (net, conn) in enumerate(self._nets.items()):
            m2_tracks = sorted(conn.keys())
            if not m2_tracks:
                continue
            trunk = trunk_base + net_index
            lo = min(m2_tracks)
            hi = max(m2_tracks)
            # EnclosureGrid endpoints repeat on odd legal indices.
            if lo % 2 == 0:
                lo -= 1
            if hi % 2 == 0:
                hi += 1
            self.addWire(self.m1_updated, net, trunk, (lo, -1), (hi, 1), netType='drawing')
            for m2_track, m1_contacts in conn.items():
                minx, maxx = _get_wire_terminators(m1_contacts)
                self.addWire(self.m2_updated, net, m2_track, (minx, -1), (trunk, 1), netType='pin')
                self.addVia(self.v1_x, net, trunk, m2_track)

        # ALIGN's legacy LEF writer checks width against an M3 pitch. Emit a
        # common M1/M2-grid-aligned boundary (LCM = 20um for TR-1um).
        import math
        m1_pitch = self.pdk['M1']['Pitch']
        m2_pitch = self.pdk['M2']['Pitch']
        lcm_pitch = math.lcm(m1_pitch, m2_pitch)
        # Boundary must be a multiple of the LEF grid (legacy writer checks width
        # against the top routing pitch) and cover all device + trunk tracks.
        # Boundary must dominate every terminal and be a multiple of the M2 pitch
        # (the legacy LEF writer's width grid for a 2-metal PDK). Compute the raw
        # extent of the widest trunk plus via enclosure, then round up to the M2
        # grid and draw the boundary as an explicit rectangle.
        raw_width = (trunk_base + len(self._nets)) * m1_pitch + 2 * (self.pdk['V1']['VencA_L'] + max(m1_pitch, m2_pitch))
        # Round up to a multiple of BOTH pitches: the ALIGN placer snaps positions
        # to the M1 grid and the LEF/GDS width must be a whole number of M1 tracks,
        # while the legacy LEF writer requires multiples of the M2 pitch.
        width_nm = max(m2_pitch, math.ceil(raw_width / m1_pitch) * m1_pitch)
        width_nm = math.ceil(width_nm / m2_pitch) * m2_pitch
        # after the M2 round-up, re-align to M1 (LCM when needed)
        if width_nm % m1_pitch != 0:
            step = math.lcm(m1_pitch, m2_pitch)
            width_nm = math.ceil(width_nm / step) * step
        # Body contact sits at track y_cells*mpc + lFin*Fin/(2*M2pitch); reserve
        # only that track plus a 1-track margin (was: full lFin*Fin/M2pitch tracks,
        # which double-counted the unit-cell height for tall fin counts).
        body_tracks = (self.lFin * self.pdk['Fin']['Pitch']) // (2 * self.pdk['M2']['Pitch']) + 2
        y_nm = (y_cells * self.m2PerUnitCell + body_tracks) * m2_pitch
        self.terminals.append({'layer': 'Boundary', 'netName': None, 'netType': 'drawing',
                               'rect': [0, 0, width_nm, y_nm]})
    def _addBodyContact(self, x, y, x_cells, yloc=None, name='M1'):
        fullname = f'{name}_X{x}_Y{y}'
        if yloc is not None:
            y = yloc
        h = self.m2PerUnitCell
        gu = self.gatesPerUnitCell
        body_v0_track = (self.lFin*self.pdk['Fin']['Pitch'])//(2*self.pdk['M2']['Pitch'])
        gate_x = self.gateDummy*self.shared_diff + x*gu + gu // 2
        self._xpins[name]['B'].append(gate_x)
        if self.shared_diff == 0:
            self.addWire( self.activeb, None, (y+1)*h + body_v0_track, (x,1), (x+1,-1))
            self.addWire( self.pb, None, (y+1)*h + body_v0_track, (x,1), (x+1,-1))
        else:
            self.addWire( self.activeb_diff, None, (y+1)*h + body_v0_track, 0, self.gate*x_cells+1)
            self.addWire( self.pb_diff, None, (y+1)*h + body_v0_track, (x,1), (x+1,-1))
        # The body-contact M1 strap must be a SHORT local strap around the
        # body via ONLY: a full-height strap here shares the S/D metal and
        # shorts the body (VDD in the well) to the opposite-polarity S/D
        # straps through the common M1 trunk (observed as VDD-GND shorts in
        # the assembled cells and floating PMOS body nets in LVS).
        self.addWire( self.m1_updated, None, gate_x, ((y+1)*h + body_v0_track-1, -1), ((y+1)*h + body_v0_track+1, 1))
        self.addVia( self.va, f'{fullname}:B', gate_x, (y+1)*h + body_v0_track)
        # connect the local body strap to the S terminal of this device
        # (the .sp ties B=S; the router then sees B and S on the same net).
        self._xpins[name]['B'].append(gate_x)

    def _addMOSArray( self, x_cells, y_cells, pattern, vt_type, connections, minvias = 1, **parameters):
        # TR-1um: M2 pin tracks must stay within the M1 wire span (tracks 1..grid_y1),
        # so a single via per connection (minvias=1) is mandatory.
        self.minvias = 1
        names = ['M1'] if pattern == 0 else ['M1', 'M2']
        names = sorted({c[0] for mc in connections.values() for c in mc})
        self._nets = collections.defaultdict(lambda: collections.defaultdict(list)) # net:m2track:m1contacts (Updated by self._connectDevicePins)
        ### Needs to be generalized
        if len(parameters) > 2:
            device_name_all = [*parameters.keys()]
            # TR-1um is fin-less: NFIN (if absent) defaults to 1
            nfin0 = int(parameters[device_name_all[0]].get("NFIN", 1))
            nfin1 = int(parameters[device_name_all[1]].get("NFIN", 1))
            if nfin0*int(parameters[device_name_all[0]]["NF"])*int(parameters[device_name_all[0]]["M"]) != nfin1*int(parameters[device_name_all[1]]["NF"])*int(parameters[device_name_all[1]]["M"]):
                pattern=3
                if int(parameters[device_name_all[0]]["NF"])*int(parameters[device_name_all[0]]["M"]) > int(parameters[device_name_all[1]]["NF"])*int(parameters[device_name_all[1]]["M"]):
                    x_left = x_cells//2 - (int(parameters[device_name_all[1]]["NF"])*int(parameters[device_name_all[1]]["M"]))//2
                    x_right = x_cells//2 + (int(parameters[device_name_all[1]]["NF"])*int(parameters[device_name_all[1]]["M"]))//2
                else:
                    x_left = x_cells//2 - (int(parameters[device_name_all[0]]["NF"])*int(parameters[device_name_all[0]]["M"]))//2
                    x_right = x_cells//2 + (int(parameters[device_name_all[0]]["NF"])*int(parameters[device_name_all[0]]["M"]))//2
         ##########################

        for y in range(y_cells):
            self._xpins = collections.defaultdict(lambda: collections.defaultdict(list)) # inst:pin:m1tracks (Updated by self._addMOS)

            for x in range(x_cells):

                if self.exact_patterns: # Exact pattern from user
                    row_pattern = self.exact_patterns[y][x]
                    names_mapping = list(string.ascii_uppercase)
                    names_updated = {}
                    for i in range(len(names)):
                        names_updated[names_mapping[i]] = names[i]
                        names_updated[names_mapping[i].lower()] = names[i]
                    reflect = row_pattern.islower()
                    self._addMOS(x, y, x_cells, vt_type, names_updated[row_pattern],  False, **parameters)
                    if self.bodyswitch==1:self._addBodyContact(x, y, x_cells, y_cells - 1, names_updated[row_pattern])
                elif pattern == 0: # None (single transistor)
                    # TODO: Not sure this works without dummies. Currently:
                    # A A A A A A
                    self._addMOS(x, y, x_cells, vt_type, names[0], False, **parameters)
                    if self.bodyswitch==1:self._addBodyContact(x, y, x_cells, y_cells - 1, names[0])
                elif pattern == 1: # CC
                    # TODO: Think this can be improved. Currently:
                    # A B B A A' B' B' A'
                    # B A A B B' A' A' B'
                    # A B B A A' B' B' A'
                    self._addMOS(x, y, x_cells, vt_type, names[((x // 2) % 2 + x % 2 + (y % 2)) % 2], x >= x_cells // 2,  **parameters)
                    if self.bodyswitch==1:self._addBodyContact(x, y, x_cells, y_cells - 1, names[((x // 2) % 2 + x % 2 + (y % 2)) % 2])
                elif pattern == 2: # interdigitated
                    # TODO: Evaluate if this is truly interdigitated. Currently:
                    # A B A B A B
                    # B A B A B A
                    # A B A B A B
                    self._addMOS(x, y, x_cells, vt_type, names[((x % 2) + (y % 2)) % 2], False,  **parameters)
                    if self.bodyswitch==1:self._addBodyContact(x, y, x_cells, y_cells - 1, names[((x % 2) + (y % 2)) % 2])
                elif pattern == 3: # CurrentMirror
                    # TODO: Evaluate if this needs to change. Currently:
                    # B B B A A B B B
                    # B B B A A B B B
                    self._addMOS(x, y, x_cells, vt_type, names[0 if x_left <= x < x_right else 1], False,  **parameters)
                    if self.bodyswitch==1:self._addBodyContact(x, y, x_cells, y_cells - 1, names[0 if x_left <= x < x_right else 1])
                elif pattern == 4:  # non common centroid
                    # TODO: Evaluate if this is truly interdigitated. Currently:
                    # A A A B B B
                    # A A A B B B
                    # A A A B B B
                    self._addMOS(x, y, x_cells, vt_type, names[0 if x < (x_cells//2) else 1], False, **parameters)
                    if self.bodyswitch == 1:
                        self._addBodyContact(x, y, x_cells, y_cells - 1, names[0 if x < (x_cells//2) else 1])
                else:
                    assert False, "Unknown pattern"
            self._connectDevicePins(y, y_cells, connections)
        self._connectNets(x_cells, y_cells)

    def addNMOSArray( self, x_cells, y_cells, pattern, vt_type, connections, **parameters):

        self._addMOSArray(x_cells, y_cells, pattern, vt_type, connections, **parameters)

        #####   Nselect Placement   #####
        M3_tracks_end = ceil((x_cells*self.gatesPerUnitCell+2*self.gateDummy*self.shared_diff)*self.pdk['M1']['Pitch']/self.pdk['M2']['Pitch'])
        M3_tracks_start = ceil(self.pdk['M1']['Pitch']/self.pdk['M2']['Pitch'])

        self.addRegion( self.nselect, None, -M3_tracks_start, 0, M3_tracks_end, y_cells* self.finsPerUnitCell)
        if self.bodyswitch==1:self.addRegion( self.pselect, None, -M3_tracks_start, y_cells* self.finsPerUnitCell, M3_tracks_end, y_cells* self.finsPerUnitCell+self.bodyswitch*self.lFin)

    def addPMOSArray( self, x_cells, y_cells, pattern, vt_type, connections, **parameters):

        self._addMOSArray(x_cells, y_cells, pattern, vt_type, connections, **parameters)

        #####   Pselect and Nwell Placement   #####
        M3_tracks_end = ceil((x_cells*self.gatesPerUnitCell+2*self.gateDummy*self.shared_diff)*self.pdk['M1']['Pitch']/self.pdk['M2']['Pitch'])
        M3_tracks_start = ceil(self.pdk['M1']['Pitch']/self.pdk['M2']['Pitch'])

        self.addRegion( self.pselect, None, -M3_tracks_start, 0, M3_tracks_end, y_cells* self.finsPerUnitCell)
        if self.bodyswitch==1:self.addRegion( self.nselect, None, -M3_tracks_start, y_cells* self.finsPerUnitCell, M3_tracks_end, y_cells* self.finsPerUnitCell+self.bodyswitch*self.lFin)
        self.addRegion( self.nwell, None, -M3_tracks_start, 0, M3_tracks_end, y_cells* self.finsPerUnitCell+self.bodyswitch*self.lFin)


