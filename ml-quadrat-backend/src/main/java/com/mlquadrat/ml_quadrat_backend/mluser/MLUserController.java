package com.mlquadrat.ml_quadrat_backend.mluser;

import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.fasterxml.jackson.databind.node.ObjectNode;

import response.JWTResponse;
@CrossOrigin
@RestController
@RequestMapping("/api/v1/user")
public class MLUserController {
	
		@Autowired
		private MLUserService service;
	
		@PostMapping(value="/login")
		public ResponseEntity<JWTResponse> login(@RequestBody MLUser user) {
			return ResponseEntity.ok(new JWTResponse(service.verify(user)));
		}
		
}
