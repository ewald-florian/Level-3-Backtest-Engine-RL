#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Phillipp
# Creation Date: -
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Parser classes. Parse snapshot and message data, i.e. convert str numbers
to int, use uniform keys, remove unnecessary message types and headers.
"""
# ---------------------------------------------------------------------------


class SnapshotParser:
    """
    Parse limit order book snapshot (A7 OB API) to proprietary format. To be 
    used as decorator @SnapshotParser.parse ahead of target method.

    :param snapshot:
        list, ...
    """

    # parse decorator . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def parse(f):        
        def wrapper(self,
                    snapshot:dict) -> dict:

            # ...
            snapshot_output = {}
            book_sides = {"Buy":1, "Sell":2}

            # ...
            for side in book_sides:
                side_dict = {}

                # ...
                for price_level in snapshot[side]:
                    side_dict[int(price_level["Price"])] = []
                    for order in price_level["Orders"]:
                        
                        # add relevant information to the respective price level
                        side_dict.get(int(price_level["Price"])).append({
                            "template_id": int(order["TemplateID"]),
                            "msg_seq_num": int(order["MsgSeqNum"]),
                            "side": int(book_sides.get(side)),
                            "price": int(price_level["Price"]),
                            "quantity": int(order["DisplayQty"]),
                            "timestamp": int(order["TrdRegTSTimePriority"])
                        })
                # add side to snapshot_output
                snapshot_output[book_sides.get(side)] = side_dict
            return f(self, snapshot_output)
        return wrapper

class MessagePacketParser:
    """
    Parse message packet (A7 EOBI API) to proprietary format. To be used as 
    decorator @MessagePacketParser.parse ahead of target method. 

    :param message_packet:
        dict, ...
    """

    # parse decorator . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def parse(f):
        def wrapper(self,
                    message_packet: list) -> list:


            message_packet_output = []        

            # parse message based on the corresponding template_id
            for message in message_packet:

                try:
                    # TODO: use match
                    # identify template_id
                    template_id = message["MessageHeader"]["TemplateID"]
                    
                    # packet header
                    if template_id == 13005:
                        continue
                    
                    # order add
                    elif template_id == 13100:
                        message = MessagePacketParser._order_add(message)
                    # order modify
                    elif template_id == 13101:
                        message = MessagePacketParser._order_modify(message)
                    # order delete
                    elif template_id == 13102:
                        message = MessagePacketParser._order_delete(message)
                    # order mass delete
                    elif template_id == 13103:
                        message = MessagePacketParser._order_mass_delete(message)
                    # full execution
                    elif template_id == 13104:
                        message = MessagePacketParser._execution_full(message)
                    # partial execution
                    elif template_id == 13105:
                        message = MessagePacketParser._execution_partial(message)
                    # order modify same priority
                    elif template_id == 13106:
                        message = MessagePacketParser._order_modify_same_priority(message)
                        
                    # trade report
                    elif template_id == 13201:
                        continue
                    # execution summary 
                    elif template_id == 13202:
                        message = MessagePacketParser._execution_summary(message)
                    
                    # product state change
                    elif template_id == 13300:
                        continue
                    # instrument state change
                    elif template_id == 13301:
                        continue
                    
                    # auction best bid/offer
                    elif template_id == 13500:  
                        continue
                    # auction clearing price
                    elif template_id == 13501:
                        continue

                    # ...
                    else:
                        print(template_id, message)
                        raise ValueError

                    # message_packet_output.append(message)
                    if message:
                        message_packet_output.append(message) 

                except Exception as error:
                    print(error)
                    raise ValueError

            return f(self, message_packet_output)
        return wrapper

    # template options . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    @staticmethod
    def _order_add(message) -> dict: # 13100

        return { 
            "template_id":      13100,
            "msg_seq_num":      int(message["MessageHeader"]["MsgSeqNum"]),
            "side":             int(message["OrderDetails"]["Side"]),
            "price":            int(message["OrderDetails"]["Price"]),
            "quantity":         int(message["OrderDetails"]["DisplayQty"]),
            "timestamp":        int(message["OrderDetails"]["TrdRegTSTimePriority"]),
            "time-in":          int(message["TrdRegTSTimeIn"])    
        }

    @staticmethod
    def _order_modify(message) -> dict: # 13101
        return { 
            "template_id":      13101,
            "msg_seq_num":      int(message["MessageHeader"]["MsgSeqNum"]),
            "side":             int(message["OrderDetails"]["Side"]),
            # ...
            "price":            int(message["OrderDetails"]["Price"]),
            "quantity":         int(message["OrderDetails"]["DisplayQty"]),
            "timestamp":        int(message["OrderDetails"]["TrdRegTSTimePriority"]),
            # ...
            "prev_price":       int(message["PrevPrice"]),
            "prev_quantity":    int(message["PrevDisplayQty"]),
            "prev_timestamp":   int(message["TrdRegTSPrevTimePriority"]),
            "time-in":          int(message["TrdRegTSTimeIn"])    
        }

    @staticmethod
    def _order_delete(message) -> dict: # 13102
        return { 
            "template_id":      13102,
            "msg_seq_num":      int(message["MessageHeader"]["MsgSeqNum"]),
            "side":             int(message["OrderDetails"]["Side"]),
            "price":            int(message["OrderDetails"]["Price"]),
            "quantity":         int(message["OrderDetails"]["DisplayQty"]),
            "timestamp":        int(message["OrderDetails"]["TrdRegTSTimePriority"]),
            "time-in":          int(message["TransactTime"])    
        }

    @staticmethod
    def _order_mass_delete(message) -> dict: # 13103
        return { 
            "template_id":      13103,
            "msg_seq_num":      int(message["MessageHeader"]["MsgSeqNum"]),
        }

    @staticmethod
    def _execution_full(message) -> dict: # 13104
        return { 
            "template_id":      13104,
            "msg_seq_num":      int(message["MessageHeader"]["MsgSeqNum"]),
            "side":             int(message["Side"]),
            "price":            int(message["Price"]),
            "quantity":         int(message["LastQty"]),
            "timestamp":        int(message["TrdRegTSTimePriority"])
        }

    @staticmethod
    def _execution_partial(message) -> dict: # 13105
        return { 
            "template_id":      13105,
            "msg_seq_num":      int(message["MessageHeader"]["MsgSeqNum"]),
            "side":             int(message["Side"]),
            "price":            int(message["Price"]),
            "quantity":         int(message["LastQty"]),
            "timestamp":        int(message["TrdRegTSTimePriority"])
        }

    @staticmethod
    def _order_modify_same_priority(message) -> dict: # 13106
        return { 
            "template_id":      13106,
            "msg_seq_num":      int(message["MessageHeader"]["MsgSeqNum"]),
            "side":             int(message["OrderDetails"]["Side"]),
            "price":            int(message["OrderDetails"]["Price"]),
            "quantity":         int(message["OrderDetails"]["DisplayQty"]),
            "timestamp":        int(message["OrderDetails"]["TrdRegTSTimePriority"]),
            "prev_price":       int(message["OrderDetails"]["Price"]),
            "prev_quantity":    int(message["PrevDisplayQty"]),
            "prev_timestamp":   int(message["OrderDetails"]["TrdRegTSTimePriority"]),
            "time-in":          int(message["TrdRegTSTimeIn"])    
        }

    @staticmethod
    def _execution_summary(message, verbose=False) -> dict: # 13202
        
        if message["AggressorTime"]:
               
               
               return { 
                    "template_id":      13202,
                    "msg_seq_num":      int(message["MessageHeader"]["MsgSeqNum"]),
                    "side":             int(message["AggressorSide"]),
                    "price":            int(message["LastPx"]),
                    "quantity":         int(message["LastQty"]),
                   # time-in like
                    "timestamp":        int(message["AggressorTime"]),
                    "exec_id":        int(message["ExecID"])
               }

        else:
            if verbose:
                print('Flawed Message: ')
                print(message)
                print(message["AggressorTime"])
                # return None
                return None
              
           

            
             


